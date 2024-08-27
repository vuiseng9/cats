OUTFILE=final_methods_mistral7B_fp32.csv
PYFILE=profile_mistral7B.py
GPUID=4
CUDA_VISIBLE_DEVICES=$GPUID python $PYFILE --ths 0.000   --filename $OUTFILE --cuths 0.000
CUDA_VISIBLE_DEVICES=$GPUID python $PYFILE --ths 0.005   --filename $OUTFILE --cuths 0.005
CUDA_VISIBLE_DEVICES=$GPUID python $PYFILE --ths 0.010   --filename $OUTFILE --cuths 0.010
CUDA_VISIBLE_DEVICES=$GPUID python $PYFILE --ths 0.030   --filename $OUTFILE --cuths 0.030
CUDA_VISIBLE_DEVICES=$GPUID python $PYFILE --ths 0.100   --filename $OUTFILE --cuths 0.100
CUDA_VISIBLE_DEVICES=$GPUID python $PYFILE --ths 0.200   --filename $OUTFILE --cuths 0.200
CUDA_VISIBLE_DEVICES=$GPUID python $PYFILE --ths 0.300   --filename $OUTFILE --cuths 0.300
CUDA_VISIBLE_DEVICES=$GPUID python $PYFILE --ths 1.250   --filename $OUTFILE --cuths 1.250
CUDA_VISIBLE_DEVICES=$GPUID python $PYFILE --ths 1.875   --filename $OUTFILE --cuths 1.875
CUDA_VISIBLE_DEVICES=$GPUID python $PYFILE --ths 2.500   --filename $OUTFILE --cuths 2.500
CUDA_VISIBLE_DEVICES=$GPUID python $PYFILE --ths 3.125   --filename $OUTFILE --cuths 3.125
CUDA_VISIBLE_DEVICES=$GPUID python $PYFILE --ths 3.750   --filename $OUTFILE --cuths 3.750
CUDA_VISIBLE_DEVICES=$GPUID python $PYFILE --ths 4.375   --filename $OUTFILE --cuths 4.375
CUDA_VISIBLE_DEVICES=$GPUID python $PYFILE --ths 5.000   --filename $OUTFILE --cuths 5.000

