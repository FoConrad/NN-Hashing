TEX_FILES = $(shell echo *.tex *.bib)
PAPER = poster
LATEX := pdflatex
TEX_FLAGS = -interaction=nonstopmode -halt-on-error

all: $(PAPER).dvi
	
$(PAPER).dvi: $(GENERATED_figures) $(TEX_FILES)
	$(LATEX) $(TEX_FLAGS) $(PAPER)
	bibtex $(PAPER)
	$(LATEX) $(TEX_FLAGS) $(PAPER)
	$(LATEX) $(TEX_FLAGS) $(PAPER)

$(PAPER).ps: $(PAPER).dvi
	dvips -t letter -o $@ $^

$(PAPER).pdf: $(PAPER).ps
	ps2pdf $^

allclean:
	rm -f *.aux *.log *.bbl *.blg *.pdf *.out

clean:
	rm -f *.aux *.log *.bbl *.blg *.pdf *.out $(PAPER).ps $(PAPER).dvi *.snm *.nav *.toc
