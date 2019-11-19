PDF=pdflatex
BIB=bibtex
LATEXOBJS= *.aux *.log *.out *.pdf *.bbl *.blg

RDIR=report
RHEADER=report
RSRCS=$(RDIR)/$(RHEADER).tex

.PHONY: clean report

report:
	$(PDF) $(RSRCS)
	$(BIB) $(RHEADER)
	$(PDF) $(RSRCS)
	$(PDF) $(RSRCS)

clean:
	rm -rf $(LATEXOBJS)
