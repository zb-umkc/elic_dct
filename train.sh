for lmbda in 1; do
    echo "Lambda: ${lmbda}"
    python3 train.py --mode train --primary_pol HH --lambda "${lmbda}" -e 200 -lr 1e-3
done