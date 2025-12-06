import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import './globals.css'
import { Providers } from './providers'
import { Navigation } from '@/components/Navigation'

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata = {
  title: 'Research Paper Explorer | Chulalongkorn University',
  description: 'Explore 19,523 academic papers with AI-powered semantic search',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className={inter.className}>
        <Providers>
          <div className="min-h-screen bg-background">
            <Navigation />
            <main>{children}</main>
            <footer className="border-t mt-20">
              <div className="container mx-auto px-4 py-8 text-center text-sm text-muted-foreground">
                <p>© 2024 Chulalongkorn University Research Observatory</p>
                <p className="mt-2">
                  Powered by AI • 19,523 Papers • 2018-2023
                </p>
              </div>
            </footer>
          </div>
        </Providers>
      </body>
    </html>
  )
}
