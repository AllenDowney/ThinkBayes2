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

