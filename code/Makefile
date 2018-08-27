FILES = Makefile \
brfss.py     euro3.py    monty2.py        sat.py          train.py \
cookie2.py   euro.py     monty.py         species.py      variability.py \
cookie.py    hockey.py   paintball.py     thinkbayes2.py  thinkstats2.py \
dice.py      kidney.py   price.py         thinkplot.py \
dungeons.py  redline_data.py  train2.py   columns.py \
euro2.py     m_and_m.py  redline.py       train3.py \


DATA = CDBRFS08.ASC.gz \
hockey_data.csv                sat_ranks.csv  showcases.2011.csv \
journal.pone.0047712.s001.csv  sat_scale.csv  showcases.2012.csv \
# BBB_data_from_Rob.csv


git:
	git add $(FILES) $(DATA)


FIGS = observed_speeds.*

all_figs:
	# TODO: update this
	python descriptive.py
	python cumulative.py
	python babyboom.py
	python continuous.py
	python hypothesis.py
	python locomotive.py
	python brfss_figs.py
	python brfss_scatter.py

FIGDEST = ../book/figs

figs:
	rsync -a $(FIGS) $(FIGDEST)


DOCS = thinkbayes2.html thinkplot.html

DOCPY = thinkbayes2.py thinkplot.py

%.html: %.py
	pydoc -w $<

.PHONY: docs $(DOCPY)

docs: $(DOCPY)

$(DOCPY):
	pydoc -w ./$@

scrub2:
	python scrub/scrub_code.py chap02soln.ipynb
	mv chap02soln_scrub.ipynb chap02.ipynb

scrub3:
	python scrub/scrub_code.py chap03soln.ipynb
	mv chap03soln_scrub.ipynb chap03.ipynb

scrub4:
	python scrub/scrub_code.py chap04soln.ipynb
	mv chap04soln_scrub.ipynb chap04.ipynb

scrub5:
	python scrub/scrub_code.py chap05soln.ipynb
	mv chap05soln_scrub.ipynb chap05.ipynb

scrub_btp01:
	python scrub/scrub_code.py btp01soln.ipynb
	mv btp01soln_scrub.ipynb btp01.ipynb

scrub9:
	python scrub/scrub_code.py chap09soln.ipynb
	mv chap09soln_scrub.ipynb chap09.ipynb

scrub7:
	python scrub/scrub_code.py chap07soln.ipynb
	mv chap07soln_scrub.ipynb chap07.ipynb

scrub_survival:
	python scrub/scrub_code.py survival_soln.ipynb
	mv survival_soln_scrub.ipynb survival.ipynb

