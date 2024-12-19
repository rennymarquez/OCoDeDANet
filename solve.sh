#/usr/bin/bash

initialk=50

for s in 12 123 1234 ; do \
  for a in 1 0.5 ; do \
    for model in GDUBNMFARD ; do \
      for instance in data_4/DataGreeneN*; do \
        echo $instance;
        ./dyncls --model $model --initial-K $initialk --matrix-seed $s --min-iters 1000 --alpha $a $instance ;
        notify.sh "$instance SEED: $s alfa: $a Model: $model ";
        sleep 5
      done
    done
  done
done

notify.sh "END";
