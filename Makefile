all: dyncls

clean:
	rm -rf build dyncls

PYTHON ?= /usr/bin/env python

dyncls: src/*.py src/*/*.py
	rm -rf build
	mkdir -p build
	for d in src src/models ; do \
		mkdir -p build/$$d ;\
		cp -pPR $$d/*.py build/$$d/ ;\
	done
	mv build/src build/dyncls
	touch -t 200001010101 build/dyncls/*.py build/dyncls/*/*.py
	mv build/dyncls/__main__.py build/
	cd build ; zip -q ../dyncls dyncls/*.py dyncls/*/*.py __main__.py
	echo '#!$(PYTHON)' > dyncls
	cat dyncls.zip >> dyncls
	rm dyncls.zip
	mv build/__main__.py build/dyncls.py
	chmod a+x dyncls

test: dyncls
	./venv/bin/python dyncls --model DBNMFARD data/Data17R0M0S1
	./venv/bin/python dyncls --model SNMF data/Data17R0M0S1
	./venv/bin/python dyncls --model SBNMF data/Data17R0M0S1
	./venv/bin/python dyncls --model SUBNMF data/Data17R0M0S1
	./venv/bin/python dyncls --model DBNMFARD --matrix-seed 1234 --initial-K 50 --alpha 1 ../dynamic-clustering/DATA/Data123S3/

