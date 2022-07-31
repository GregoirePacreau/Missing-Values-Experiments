python main.py --dim_size=100 --sample_size=200 --n_exp=1 --effective_rank=25 --method=TSGS --contamination=adversarial
python main.py --dim_size=100 --sample_size=200 --n_exp=1 --effective_rank=25 --method=TSGS --contamination=bernoulli
python main.py --dim_size=100 --sample_size=200 --n_exp=1 --effective_rank=25 --method=random_MV --contamination=adversarial
python main.py --dim_size=100 --sample_size=200 --n_exp=1 --effective_rank=25 --method=DDC_MV --contamination=adversarial
python main.py --dim_size=100 --sample_size=200 --n_exp=1 --effective_rank=25 --method=tail_MV --contamination=adversarial
python main.py --dim_size=100 --sample_size=200 --n_exp=1 --effective_rank=25 --method=DI --contamination=adversarial
