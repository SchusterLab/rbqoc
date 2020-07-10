PDF=pdflatex
BIB=bibtex
LATEXOBJS= *.aux *.log *.out *.pdf *.bbl *.blg

PHEADER=paper
PSRCS=$(PHEADER).tex

.PHONY: all clean

all: paper

paper:
	$(PDF) $(PSRCS)
	$(BIB) $(PHEADER)
	$(PDF) $(PSRCS)
	$(PDF) $(PSRCS)

clean:
	rm -rf $(LATEXOBJS)
