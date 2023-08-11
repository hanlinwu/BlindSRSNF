echo "Model: $1"

python test_diff.py --checkpoint $1 --lambda1 1.2 --lambda2 2.4 --theta 0 --noise 0 --datasets sr-geo-15,Google-15 --gpu 0
python test_diff.py --checkpoint $1 --lambda1 1.2 --lambda2 2.4 --theta 0 --noise 5 --datasets sr-geo-15,Google-15 --gpu 0
python test_diff.py --checkpoint $1 --lambda1 1.2 --lambda2 2.4 --theta 0 --noise 10 --datasets sr-geo-15,Google-15 --gpu 0

python test_diff.py --checkpoint $1 --lambda1 1.2 --lambda2 2.4 --theta 45 --noise 0 --datasets sr-geo-15,Google-15 --gpu 0
python test_diff.py --checkpoint $1 --lambda1 1.2 --lambda2 2.4 --theta 45 --noise 5 --datasets sr-geo-15,Google-15 --gpu 0
python test_diff.py --checkpoint $1 --lambda1 1.2 --lambda2 2.4 --theta 45 --noise 10 --datasets sr-geo-15,Google-15 --gpu 0

python test_diff.py --checkpoint $1 --lambda1 2.4 --lambda2 1.2 --theta 45 --noise 0 --datasets sr-geo-15,Google-15 --gpu 0
python test_diff.py --checkpoint $1 --lambda1 2.4 --lambda2 1.2 --theta 45 --noise 5 --datasets sr-geo-15,Google-15 --gpu 0
python test_diff.py --checkpoint $1 --lambda1 2.4 --lambda2 1.2 --theta 45 --noise 10 --datasets sr-geo-15,Google-15 --gpu 0

python test_diff.py --checkpoint $1 --lambda1 3.6 --lambda2 2.4 --theta 0 --noise 0 --datasets sr-geo-15,Google-15 --gpu 0
python test_diff.py --checkpoint $1 --lambda1 3.6 --lambda2 2.4 --theta 0 --noise 5 --datasets sr-geo-15,Google-15 --gpu 0
python test_diff.py --checkpoint $1 --lambda1 3.6 --lambda2 2.4 --theta 0 --noise 10 --datasets sr-geo-15,Google-15 --gpu 0