"""
Literature Review Assistant - Streamlit Dashboard
Academic research paper search with AI-powered analysis

Pipeline: Part 1 (Data) → Part 2 (Embeddings) → Part 3 (Network) → Part 4 (Dashboard)
"""

import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
from pathlib import Path
import asyncio
import time
import plotly.graph_objects as go
import plotly.express as px
from sentence_transformers import SentenceTransformer
import chromadb
from sklearn.metrics.pairwise import cosine_similarity
import json
from anthropic import AsyncAnthropic

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Research Paper Search",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern, clean CSS
st.markdown("""
<style>
    /* Main layout */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* Headers */
    h1 {
        color: #1e3a8a;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }

    h2 {
        color: #3b82f6;
        font-weight: 600;
        margin-top: 2rem;
    }

    h3 {
        color: #64748b;
        font-weight: 600;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1.5rem;
        background-color: #f8fafc;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
    }

    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        padding: 0 1.5rem;
        font-size: 1rem;
        font-weight: 500;
        color: #64748b;
    }

    .stTabs [aria-selected="true"] {
        background-color: white;
        color: #3b82f6;
        border-radius: 0.375rem;
    }

    /* Cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 0.75rem;
        color: white;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
    }

    /* Expander */
    .streamlit-expanderHeader {
        font-size: 1rem;
        font-weight: 500;
        color: #1e293b;
    }

    /* Sidebar */
    .css-1d391kg {
        background-color: #f8fafc;
    }

    /* Info boxes */
    .stAlert {
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE
# ============================================================================

if 'papers' not in st.session_state:
    st.session_state.papers = None
if 'G' not in st.session_state:
    st.session_state.G = None
if 'search_performed' not in st.session_state:
    st.session_state.search_performed = False

# ============================================================================
# DATA LOADING & CACHING
# ============================================================================

SUMMARY_CACHE_FILE = Path('../data/abstract_summaries.json')

@st.cache_resource
def load_data():
    """Load papers, embeddings, and ML models"""
    df = pd.read_parquet('../data/processed/papers.parquet')
    embeddings = np.load('../data/embeddings/paper_embeddings.npy')
    model = SentenceTransformer('all-MiniLM-L6-v2')
    client = chromadb.PersistentClient(path="../data/vector_db")
    collection = client.get_collection("papers")
    return df, embeddings, model, collection

@st.cache_data
def load_subject_hierarchy():
    """Load subject area hierarchy for color mapping"""
    with open('../data/processed/subject_hierarchy.json', 'r') as f:
        return json.load(f)['subject_groups']

@st.cache_data
def load_treemap_data():
    """Load pre-computed tree map data (generated in Part 1)"""
    with open('../data/processed/treemap_data.json', 'r') as f:
        return json.load(f)

def load_summary_cache():
    """Load AI-generated summaries cache"""
    if SUMMARY_CACHE_FILE.exists():
        with open(SUMMARY_CACHE_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_summary_cache(cache):
    """Save summaries to disk cache"""
    SUMMARY_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(SUMMARY_CACHE_FILE, 'w') as f:
        json.dump(cache, f, indent=2)

# ============================================================================
# COLOR SCHEME
# ============================================================================

CATEGORY_COLORS = {
    'Medicine & Health': '#E91E63',
    'Life Sciences': '#4CAF50',
    'Computer Science & AI': '#2196F3',
    'Engineering': '#FF9800',
    'Materials & Chemistry': '#9C27B0',
    'Physics': '#00BCD4',
    'Environmental Science': '#8BC34A',
    'Other': '#9E9E9E'
}

STANCE_COLORS = {
    'SUPPORT': '#10b981',
    'CONTRADICT': '#ef4444',
    'NEUTRAL': '#6b7280'
}

def get_subject_color(subject, subject_groups):
    """Get color for a subject area based on its category"""
    for category, keywords in subject_groups.items():
        if category == 'Other':
            continue
        for keyword in keywords:
            if keyword.lower() in subject.lower():
                return CATEGORY_COLORS[category]
    return CATEGORY_COLORS['Other']

# ============================================================================
# SEARCH & AI ANALYSIS
# ============================================================================

def smart_search(query, model, collection, df, num_papers=20, min_threshold=35.0):
    """Semantic search with smart threshold"""
    query_emb = model.encode(query)

    results = collection.query(
        query_embeddings=[query_emb.tolist()],
        n_results=min(100, len(df))
    )

    papers_data = []
    for i, (meta, distance) in enumerate(zip(results['metadatas'][0], results['distances'][0])):
        similarity = (2.0 - distance) / 2.0 * 100
        papers_data.append({
            'rank': i + 1,
            'title': meta['title'],
            'similarity': similarity,
            'distance': distance
        })

    results_df = pd.DataFrame(papers_data)
    results_df = results_df.merge(
        df[['title', 'id', 'abstract', 'year', 'citation_count', 'authors', 'subject_areas']],
        on='title',
        how='left'
    )

    title_to_idx = {title: idx for idx, title in enumerate(df['title'])}
    results_df['paper_idx'] = results_df['title'].map(title_to_idx)

    above_threshold = results_df[results_df['similarity'] >= min_threshold]

    if len(above_threshold) >= num_papers:
        return above_threshold.head(num_papers).reset_index(drop=True)
    else:
        return results_df.head(num_papers).reset_index(drop=True)

async def generate_summary(paper, client):
    """Generate one-sentence summary (cached)"""
    prompt = f"""Summarize in one sentence:

{paper['abstract'][:800]}

Summary:"""

    try:
        response = await client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=80,
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )
        return {'id': paper['id'], 'summary': response.content[0].text.strip()}
    except Exception as e:
        return {'id': paper['id'], 'summary': f"Summary of: {paper['title'][:100]}"}

async def detect_stance(paper, query, client):
    """Detect paper stance relative to query"""
    prompt = f"""Query: {query}

Abstract: {paper['abstract'][:600]}

Does this paper SUPPORT, CONTRADICT, or is NEUTRAL to the query? One word only."""

    try:
        response = await client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=5,
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )

        text = response.content[0].text.strip().upper()
        if 'SUPPORT' in text:
            return {'id': paper['id'], 'stance': 'SUPPORT'}
        elif 'CONTRADICT' in text:
            return {'id': paper['id'], 'stance': 'CONTRADICT'}
        else:
            return {'id': paper['id'], 'stance': 'NEUTRAL'}
    except Exception as e:
        return {'id': paper['id'], 'stance': 'NEUTRAL'}

async def analyze_papers(papers, query, api_key):
    """AI analysis: summaries (cached) + stance (fresh)"""
    client = AsyncAnthropic(api_key=api_key)
    summary_cache = load_summary_cache()
    results = {}

    # Get summaries (use cache when possible)
    summary_tasks = []
    papers_need_summary = []

    for _, paper in papers.iterrows():
        paper_id = str(paper['id'])

        if paper_id in summary_cache and not summary_cache[paper_id].startswith("Error"):
            results[paper_id] = {'id': paper_id, 'summary': summary_cache[paper_id]}
        else:
            papers_need_summary.append(paper)
            summary_tasks.append(generate_summary(paper, client))

    if summary_tasks:
        summary_results = await asyncio.gather(*summary_tasks)
        for result in summary_results:
            paper_id = str(result['id'])
            results[paper_id] = {'id': paper_id, 'summary': result['summary']}
            summary_cache[paper_id] = result['summary']
        save_summary_cache(summary_cache)

    # Detect stance (always fresh, query-dependent)
    stance_tasks = [detect_stance(paper, query, client) for _, paper in papers.iterrows()]
    stance_results = await asyncio.gather(*stance_tasks)

    for result in stance_results:
        paper_id = str(result['id'])
        results[paper_id]['stance'] = result['stance']

    return results

def build_network(papers_df, embeddings_full, threshold=0.60, max_edges=5):
    """Build similarity network"""
    paper_embeddings = embeddings_full[papers_df['paper_idx'].values]
    sim_matrix = cosine_similarity(paper_embeddings)

    G = nx.Graph()

    for idx, row in papers_df.iterrows():
        G.add_node(
            row['id'],
            title=row['title'],
            year=row['year'],
            citation_count=row['citation_count']
        )

    paper_ids = papers_df['id'].values

    for i in range(len(papers_df)):
        sims = sim_matrix[i].copy()
        sims[i] = -1
        top_k = np.argsort(sims)[::-1][:max_edges]

        for j in top_k:
            if sims[j] >= threshold:
                G.add_edge(paper_ids[i], paper_ids[j], similarity=float(sims[j] * 100))

    # Detect communities
    if G.number_of_edges() > 0:
        from networkx.algorithms import community
        communities = list(community.greedy_modularity_communities(G))
        for i, comm in enumerate(communities):
            for node in comm:
                G.nodes[node]['community'] = i
    else:
        for node in G.nodes():
            G.nodes[node]['community'] = 0

    return G

# ============================================================================
# UI COMPONENTS
# ============================================================================

def show_home_page():
    """Display home page with tree map"""
    st.title("📚 Research Paper Explorer")
    st.markdown("### Explore 19,523 academic papers across multiple disciplines")

    # Load all papers
    @st.cache_data
    def load_all_papers():
        return pd.read_parquet('../data/processed/papers.parquet')

    df = load_all_papers()

    # Dataset metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("📄 Total Papers", f"{len(df):,}")
    with col2:
        st.metric("📅 Year Range", f"{df['year'].min()}-{df['year'].max()}")
    with col3:
        st.metric("📊 Total Citations", f"{int(df['citation_count'].sum()):,}")
    with col4:
        unique_subjects = df['subject_areas'].explode().nunique()
        st.metric("🏷️ Subject Areas", unique_subjects)

    st.markdown("---")

    # Tree map
    st.markdown("## 🗺️ Research Landscape")
    st.markdown("Interactive visualization of papers by subject area")

    with st.spinner("Loading visualization..."):
        treemap_data = load_treemap_data()

    fig = go.Figure(go.Treemap(
        labels=treemap_data['labels'],
        parents=treemap_data['parents'],
        values=treemap_data['values'],
        marker=dict(
            colors=treemap_data['colors'],
            line=dict(width=2, color='white')
        ),
        textposition='middle center',
        textfont=dict(size=13, color='white', family='Arial'),
        hovertemplate='<b>%{label}</b><br>Papers: %{value}<extra></extra>'
    ))

    fig.update_layout(
        height=650,
        margin=dict(t=20, l=10, r=10, b=10),
        font=dict(size=12)
    )

    st.plotly_chart(fig, use_container_width=True)

    # Top subjects table
    st.markdown("---")
    st.markdown("### 🏆 Top 20 Subject Areas")

    subject_list = []
    for subjects in df['subject_areas']:
        if hasattr(subjects, '__iter__') and not isinstance(subjects, str):
            subject_list.extend(subjects)

    subject_counts = pd.DataFrame({
        'Subject Area': pd.Series(subject_list).value_counts().head(20).index,
        'Papers': pd.Series(subject_list).value_counts().head(20).values
    })
    subject_counts['% of Total'] = (subject_counts['Papers'] / len(df) * 100).round(1)

    st.dataframe(subject_counts, hide_index=True, use_container_width=True)

def show_paper_card(row, subject_groups):
    """Display individual paper card"""
    # Stance badge
    stance = row['stance']
    color = STANCE_COLORS[stance]
    emoji = {'SUPPORT': '✓', 'CONTRADICT': '✗', 'NEUTRAL': '○'}[stance]

    with st.expander(f"**{emoji} {stance}** | {row['title']}", expanded=False):
        # Summary
        st.markdown(f"*{row['summary']}*")
        st.markdown("")

        # Subject tags
        if 'subject_areas' in row:
            subjects = row['subject_areas']
            if hasattr(subjects, '__iter__') and not isinstance(subjects, str) and len(subjects) > 0:
                subject_html = "**Topics:** "
                for i, subject in enumerate(subjects[:5]):
                    bg_color = get_subject_color(subject, subject_groups)
                    subject_html += f'<span style="background:{bg_color}; color:white; padding:2px 8px; margin:2px; border-radius:4px; font-size:0.85em; display:inline-block;">{subject}</span> '

                if len(subjects) > 5:
                    subject_html += f'<span style="color:#6b7280; font-size:0.85em;">+{len(subjects)-5} more</span>'

                st.markdown(subject_html, unsafe_allow_html=True)
                st.markdown("")

        # Metadata
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.caption(f"📊 **Relevance:** {row['similarity']:.1f}%")
        with col2:
            st.caption(f"📝 **Citations:** {row['citation_count']}")
        with col3:
            st.caption(f"📅 **Year:** {row['year']}")
        with col4:
            st.caption(f"**Stance:** {emoji} {stance}")

        # Full details toggle
        if st.checkbox(f"📄 View Abstract", key=f"abstract_{row['id']}"):
            st.markdown("**Abstract:**")
            st.markdown(row['abstract'])
            st.caption(f"**Authors:** {row['authors']}")

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    # Header
    st.title("📚 Literature Review Assistant")
    st.markdown("AI-powered research paper search and analysis")

    # Sidebar
    with st.sidebar:
        st.header("🔍 Search")

        query = st.text_input(
            "Research Question:",
            value="machine learning for medical diagnosis",
            help="Enter your research topic or question"
        )

        num_papers = st.slider(
            "Number of Papers:",
            min_value=5,
            max_value=50,
            value=20,
            help="How many papers to retrieve"
        )

        threshold = st.slider(
            "Network Threshold:",
            min_value=0.5,
            max_value=0.8,
            value=0.6,
            step=0.05,
            help="Similarity threshold for network edges"
        )

        st.markdown("---")
        st.subheader("🔑 API Key")
        api_key = st.text_input(
            "Anthropic API Key:",
            type="password",
            value="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",
            help="Required for AI analysis"
        )

        st.markdown("---")
        search_btn = st.button("🚀 Search & Analyze", type="primary", use_container_width=True)

        if search_btn and not api_key:
            st.error("⚠️ Please enter your API key")

    # Load data
    with st.spinner("Loading research database..."):
        df, embeddings, model, collection = load_data()

    # Search
    if search_btn and api_key:
        with st.spinner(f"Searching for '{query}'..."):
            progress = st.progress(0)
            status = st.empty()

            # Vector search
            status.text("🔍 Searching 19K papers...")
            papers = smart_search(query, model, collection, df, num_papers)
            progress.progress(25)

            # AI analysis
            status.text(f"🤖 Analyzing {len(papers)} papers with Claude...")
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            analysis = loop.run_until_complete(analyze_papers(papers, query, api_key))
            loop.close()
            progress.progress(60)

            # Add results
            papers['summary'] = papers['id'].map(lambda x: analysis[x]['summary'])
            papers['stance'] = papers['id'].map(lambda x: analysis[x]['stance'])
            progress.progress(80)

            # Build network
            status.text("🕸️ Building similarity network...")
            G = build_network(papers, embeddings, threshold=threshold)
            progress.progress(100)

            # Store in session
            st.session_state.papers = papers
            st.session_state.G = G
            st.session_state.search_performed = True

            progress.empty()
            status.empty()
            st.success(f"✓ Found {len(papers)} papers")

    # Create tabs
    if st.session_state.search_performed and st.session_state.papers is not None:
        tabs = st.tabs(["🏠 Home", "📈 Timeline", "📄 Papers"])
    else:
        tabs = [st.tabs(["🏠 Home"])[0], None, None]

    # Tab: Home
    with tabs[0]:
        show_home_page()

    # Tab: Timeline
    if tabs[1] is not None:
        papers = st.session_state.papers

        with tabs[1]:
            st.subheader("📈 Temporal Analysis")

            # Year filter
            year_range = st.slider(
                "Year Range:",
                min_value=int(papers['year'].min()),
                max_value=int(papers['year'].max()),
                value=(int(papers['year'].min()), int(papers['year'].max()))
            )

            filtered = papers[(papers['year'] >= year_range[0]) & (papers['year'] <= year_range[1])]

            # Charts
            col1, col2 = st.columns(2)

            with col1:
                # Papers over time
                year_counts = filtered['year'].value_counts().sort_index()
                fig1 = px.bar(
                    x=year_counts.index,
                    y=year_counts.values,
                    title="📅 Papers Over Time",
                    labels={'x': 'Year', 'y': 'Papers'}
                )
                fig1.update_layout(showlegend=False)
                st.plotly_chart(fig1, use_container_width=True)

            with col2:
                # Stance distribution
                stance_counts = papers['stance'].value_counts()
                fig2 = px.pie(
                    values=stance_counts.values,
                    names=stance_counts.index,
                    title="🎯 Stance Distribution",
                    color=stance_counts.index,
                    color_discrete_map=STANCE_COLORS,
                    hole=0.4
                )
                st.plotly_chart(fig2, use_container_width=True)

            # Stance by year
            st.markdown("### Stance Distribution by Year")
            stance_by_year = filtered.groupby(['year', 'stance']).size().unstack(fill_value=0)
            fig3 = px.bar(
                stance_by_year,
                title="",
                color_discrete_map=STANCE_COLORS,
                barmode='stack'
            )
            fig3.update_layout(xaxis_title="Year", yaxis_title="Papers", showlegend=True)
            st.plotly_chart(fig3, use_container_width=True)

    # Tab: Papers
    if tabs[2] is not None:
        papers = st.session_state.papers
        subject_groups = load_subject_hierarchy()

        with tabs[2]:
            st.subheader("📄 Paper Details")

            # Filters
            col1, col2, col3 = st.columns(3)

            with col1:
                stance_filter = st.multiselect(
                    "Stance:",
                    options=['All', 'SUPPORT', 'NEUTRAL', 'CONTRADICT'],
                    default=['All']
                )

            with col2:
                # Subject filter
                all_subjects = []
                if 'subject_areas' in papers.columns:
                    for subjects in papers['subject_areas']:
                        if hasattr(subjects, '__iter__') and not isinstance(subjects, str):
                            all_subjects.extend(subjects)
                unique_subjects = sorted(list(set(all_subjects)))

                subject_filter = st.multiselect(
                    "Subject Area:",
                    options=['All'] + unique_subjects,
                    default=['All']
                )

            with col3:
                sort_by = st.selectbox(
                    "Sort by:",
                    options=['Relevance', 'Citations', 'Year'],
                    index=0
                )

            # Apply filters
            papers_display = papers.copy()

            if 'All' not in stance_filter and len(stance_filter) > 0:
                papers_display = papers_display[papers_display['stance'].isin(stance_filter)]

            if 'All' not in subject_filter and len(subject_filter) > 0:
                def has_subject(subjects):
                    if not hasattr(subjects, '__iter__') or isinstance(subjects, str):
                        return False
                    return any(s in subject_filter for s in subjects)
                papers_display = papers_display[papers_display['subject_areas'].apply(has_subject)]

            # Sort
            if sort_by == 'Relevance':
                papers_display = papers_display.sort_values('similarity', ascending=False)
            elif sort_by == 'Citations':
                papers_display = papers_display.sort_values('citation_count', ascending=False)
            else:
                papers_display = papers_display.sort_values('year', ascending=False)

            st.write(f"Showing **{len(papers_display)}** papers")

            # Display papers
            for _, row in papers_display.iterrows():
                show_paper_card(row, subject_groups)

            # Export
            st.markdown("---")
            csv = papers_display.to_csv(index=False)
            st.download_button(
                label="📥 Download Results (CSV)",
                data=csv,
                file_name=f"papers_{query[:30].replace(' ', '_')}.csv",
                mime="text/csv",
                use_container_width=True
            )

if __name__ == "__main__":
    main()
