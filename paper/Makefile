PDF=pdflatex
BIB=bibtex
LATEXOBJS= *.aux *.log *.out *.pdf *.bbl *.blg mainNotes.bib

PHEADER=main
PSRCS=$(PHEADER).tex

.PHONY: all clean

all: paper

paper:
	$(PDF) $(PSRCS)
	$(BIB) $(PHEADER)
	$(PDF) $(PSRCS)
	$(PDF) $(PSRCS)

paper-no-bib:
	$(PDF) $(PSRCS)

clean:
	rm -rf $(LATEXOBJS)
