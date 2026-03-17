#!/bin/bash
echo "Compiling paper..."
pdflatex -interaction=nonstopmode paper_arxiv.tex
bibtex paper_arxiv
pdflatex -interaction=nonstopmode paper_arxiv.tex
pdflatex -interaction=nonstopmode paper_arxiv.tex
echo ""
echo "Done. Output: paper_arxiv.pdf"
echo "Check for errors above."
