F=thinkbayes2

PDFFLAGS = -dCompatibilityLevel=1.4 -dPDFSETTINGS=/prepress \
           -dCompressPages=true -dUseFlateCompression=true  \
           -dEmbedAllFonts=true -dSubsetFonts=true -dMaxSubsetPct=100


all:	book.tex
	pdflatex book
	makeindex book.idx
	pdflatex book
	mv book.pdf thinkbayes2.pdf
	evince thinkbayes2.pdf

plastex:
	# Before running plastex, we need the current directory in PYTHONPATH
	# export PYTHONPATH=$PYTHONPATH:.
	cp book.tex $(F).tex
	python2 preprocess.py $(F).tex > $(F).plastex
	plastex --renderer=DocBook --theme=book --image-resolution=300 --filename=$(F).xml $(F).plastex
	cd $(F); python2 ../postprocess.py $(F).xml > temp; mv temp $(F).xml

xxe:
	xmlcopyeditor ~/thinkbayes2/book/book/book.xml &

lint:
	xmllint -noout book/book.xml

OREILLY = atlas

oreilly:
	rsync -a book/ $(OREILLY)
	rsync -a figs/* $(OREILLY)/figs/
	cd $(OREILLY); git add .
	cd $(OREILLY); git commit -m "Automated check in."
	cd $(OREILLY); git push

clean:
	rm -f *~ *.aux *.log *.dvi *.idx *.ilg *.ind *.toc
HEVEABUILDLOG=build.log

# if a bug (in ocaml?) causes "make hevea" to fail; use "make -i hevea" instead

.PHONY: hevea
hevea:
	sed 's/\(figs\/[^.]*\).\(pdf\|png\)/\1.eps/' book.tex > thinkbayes2.tex
	rm -rf html
	mkdir html
	hevea macros.hva -O -e latexonly htmlonly thinkbayes2 2>&1 | tee ${HEVEABUILDLOG}
# the following greps are a kludge to prevent imagen from seeing
# the definitions in latexonly, and to avoid headers on the images
	grep -v latexonly thinkbayes2.image.tex > a; mv a thinkbayes2.image.tex
	sed s/\\\\usepackage{fancyhdr}// < thinkbayes2.image.tex > a; mv a thinkbayes2.image.tex
	imagen -png thinkbayes2
	hacha thinkbayes2.html
	cp up.png next.png back.png html
	cp custom.css thinkbayes2.css
	mv index.html thinkbayes2.css thinkbayes2*.html thinkbayes2*.png *motif.gif html

DEST = /home/downey/public_html/greent/thinkbayes2

distrib:
	rm -rf dist
	mkdir dist
	rsync -a thinkbayes2.pdf html dist
	rsync -a dist/* $(DEST)
	chmod -R o+r $(DEST)/*
	cd $(DEST)/..; sh back
