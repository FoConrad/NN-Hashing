TEX_FILES = $(shell echo *.tex *.bib)
PAPER = proposal
LATEX := pdflatex
TEX_FLAGS = -interaction=nonstopmode -halt-on-error

all: proposal.dvi
	
proposal.dvi: $(GENERATED_figures) $(TEX_FILES)
	$(LATEX) $(TEX_FLAGS) proposal
	bibtex proposal
	$(LATEX) $(TEX_FLAGS) proposal
	$(LATEX) $(TEX_FLAGS) proposal

proposal.ps: proposal.dvi
	dvips -t letter -o $@ $^

proposal.pdf: proposal.ps
	ps2pdf $^

allclean:
	rm -f *.aux *.log *.bbl *.blg *.pdf *.out

clean:
	rm -f *.aux *.log *.bbl *.blg *.pdf *.out proposal.ps proposal.dvi 
