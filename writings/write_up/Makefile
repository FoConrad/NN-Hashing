TEX_FILES = $(shell echo *.tex *.bib)
PAPER = write_up
LATEX := pdflatex
TEX_FLAGS = -interaction=nonstopmode -halt-on-error

all: write_up.dvi
	
write_up.dvi: $(GENERATED_figures) $(TEX_FILES)
	$(LATEX) $(TEX_FLAGS) write_up
	bibtex write_up
	$(LATEX) $(TEX_FLAGS) write_up
	$(LATEX) $(TEX_FLAGS) write_up

write_up.ps: write_up.dvi
	dvips -t letter -o $@ $^

write_up.pdf: write_up.ps
	ps2pdf $^

allclean:
	rm -f *.aux *.log *.bbl *.blg *.pdf *.out

clean:
	rm -f *.aux *.log *.bbl *.blg *.pdf *.out write_up.ps write_up.dvi 
