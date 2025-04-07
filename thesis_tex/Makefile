all: thesis.pdf abstract-cs.pdf abstract-en.pdf

%.pdf: force
	latexmk $*.tex

clean:
	rm -rf tmp
	latexmk -C

.PHONY: force all clean
