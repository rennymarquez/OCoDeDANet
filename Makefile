all: ocodedanet

clean:
	rm -rf build ocodedanet

PYTHON ?= /usr/bin/env python

ocodedanet: src/*.py src/*/*.py
	rm -rf build
	mkdir -p build
	for d in src src/models ; do \
		mkdir -p build/$$d ;\
		cp -pPR $$d/*.py build/$$d/ ;\
	done
	mv build/src build/ocodedanet
	touch -t 200001010101 build/ocodedanet/*.py build/ocodedanet/*/*.py
	mv build/ocodedanet/__main__.py build/
	cd build ; zip -q ../ocodedanet ocodedanet/*.py ocodedanet/*/*.py __main__.py
	echo '#!$(PYTHON)' > ocodedanet
	cat ocodedanet.zip >> ocodedanet
	rm ocodedanet.zip
	mv build/__main__.py build/ocodedanet.py
	chmod a+x ocodedanet

test: ocodedanet
	./venv/bin/python ocodedanet --model DBNMFARD data/Data17R0M0S1
	./venv/bin/python ocodedanet --model SNMF data/Data17R0M0S1
	./venv/bin/python ocodedanet --model SBNMF data/Data17R0M0S1
	./venv/bin/python ocodedanet --model SUBNMF data/Data17R0M0S1
	./venv/bin/python ocodedanet --model DBNMFARD --matrix-seed 1234 --initial-K 50 --alpha 1 ../dynamic-clustering/DATA/Data123S3/

