from joblib import Parallel, delayed

from models.utils import *

def main():
    parser = setup_parser()
    args = parser.parse_args()    

    BINARY_MODEL: bool = args.binary
    PROCESSES: int = cpu_count() if args.processes < 0 else cpu_count()-1 if args.processes > cpu_count() or args.processes == 0 else args.processes
    RUN_MODE: str = args.run_mode
    PARALLEL: bool = args.parallel
    TIMEOUT = 99999

    # load data
    print(f'[{datetime.now().isoformat()}] Loading Data...')
    job_df = load_data()
    X, hint_l, opt_l, targets_l = prepare_data(job_df, './data', f'sql_embedding_comb_', BINARY_MODEL)

    # generate experiments
    experiments = generate_experiments(BINARY_MODEL, RUN_MODE, PARALLEL, X, hint_l, opt_l, targets_l, k=10)

    # run experiments
    if PARALLEL:
        print(f'[{datetime.now().isoformat()}] Beginning experiments across {PROCESSES} CPU cores and 19 GPU cores in run mode {RUN_MODE}')
        parallel = Parallel(
            n_jobs=PROCESSES,
            timeout=TIMEOUT,
            backend='loky',
            prefer='processes',
        )
        results = parallel(delayed(train_model)(idx+1,BINARY_MODEL,**experiment) for idx, experiment in enumerate(experiments))
    else:
        print(f'[{datetime.now().isoformat()}] Beginning experiments across in run mode {RUN_MODE}')
        results = [train_model(idx+1, BINARY_MODEL, **experiment) for idx, experiment in enumerate(experiments)]

    # save results and exit
    print(f'[{datetime.now().isoformat()}] Experiments Completed - Saving Data\n')
    torch.save(results, f"./results/stats_qp_BINARY_{datetime.now().date().strftime('%Y%m%d')}_model_results.pt")

    # exiting
    print(f'[{datetime.now().isoformat()}] Exiting Successfully')
    exit(0)


if __name__ == '__main__':
    main()