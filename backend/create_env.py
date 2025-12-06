#!/usr/bin/env python3
"""
Create .env file with proper UTF-8 encoding
"""

env_content = """ANTHROPIC_API_KEY=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
"""

# Write with UTF-8 encoding (no BOM)
with open('.env', 'w', encoding='utf-8') as f:
    f.write(env_content.strip())

print(".env file created successfully with UTF-8 encoding")
