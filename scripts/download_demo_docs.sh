#!/bin/bash
cd ~/projects/rag-system/data/demo_docs

echo "Downloading sample documents for demos..."

# 1. Legal - Sample contracts (public domain)
echo "📄 Downloading legal contracts..."
mkdir -p legal
curl -sL "https://www.sec.gov/Archives/edgar/data/1318605/000156459021004599/tsla-ex101_14.htm" -o legal/tesla_employment_agreement.html 2>/dev/null

# 2. Technical Documentation - Public APIs
echo "📄 Downloading technical docs..."
mkdir -p technical
curl -sL "https://raw.githubusercontent.com/fastapi/fastapi/master/README.md" -o technical/fastapi_readme.md
curl -sL "https://raw.githubusercontent.com/langchain-ai/langchain/master/README.md" -o technical/langchain_readme.md
curl -sL "https://raw.githubusercontent.com/run-llama/llama_index/main/README.md" -o technical/llamaindex_readme.md

# 3. Research Papers - arXiv (public)
echo "📄 Downloading research papers..."
mkdir -p research
curl -sL "https://arxiv.org/pdf/1706.03762.pdf" -o research/attention_is_all_you_need.pdf 2>/dev/null
curl -sL "https://arxiv.org/pdf/2005.11401.pdf" -o research/rag_paper.pdf 2>/dev/null

# 4. Company policies - Sample employee handbook (public templates)
echo "📄 Creating sample HR docs..."
mkdir -p hr

echo "Done! Documents saved to data/demo_docs/"
ls -la */
