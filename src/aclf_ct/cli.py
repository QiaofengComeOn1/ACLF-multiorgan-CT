"""Command-line interface for ACLF analyses."""
import argparse


def main(argv=None):
    parser = argparse.ArgumentParser(description='ACLF multi-organ CT analyses')
    sub = parser.add_subparsers(dest='command', required=True)
    evaluate = sub.add_parser(
        'evaluate',
        help='Evaluate saved prediction models'
    )
    evaluate.add_argument('--runs-root', required=True)
    evaluate.add_argument('--out', required=True)
    evaluate.add_argument('--trust-model-files', action='store_true')
    fixed = sub.add_parser('evaluate-probabilities', help='Evaluate fixed y_true/p_aclf CSV')
    fixed.add_argument('--input', required=True)
    fixed.add_argument('--threshold', type=float, default=.5)
    fixed.add_argument('--out', required=True)
    fit = sub.add_parser('train', help='Refit selected classifier from frozen selected features')
    fit.add_argument('--training-csv', required=True)
    fit.add_argument(
        '--model',
        required=True,
        choices=['ClinicalMP','Liver','Spleen','Muscle','LSMFusion','CLSMFusion']
    )
    fit.add_argument(
        '--benchmark',
        action='store_true',
        help='Compare all configured candidate classifiers'
    )
    fit.add_argument('--out', required=True)
    fit.add_argument(
        '--selection-dir',
        help='Directory containing the selected matrices and preprocessing metadata'
    )
    predict = sub.add_parser('predict', help='Apply a newly fitted bundle to both test cohorts')
    predict.add_argument('--bundle', required=True)
    predict.add_argument('--internal', required=True)
    predict.add_argument('--external', required=True)
    predict.add_argument('--trust-model-files', action='store_true')
    predict.add_argument('--out', required=True)
    prepare = sub.add_parser(
        'prepare-selected',
        help='Fit scaling parameters to selected training features'
    )
    prepare.add_argument('--training', required=True)
    prepare.add_argument('--internal', required=True)
    prepare.add_argument('--external', required=True)
    prepare.add_argument(
        '--model',
        required=True,
        choices=['ClinicalMP','Liver','Spleen','Muscle','LSMFusion','CLSMFusion']
    )
    prepare.add_argument('--out', required=True)
    fuse = sub.add_parser(
        'fuse',
        help='Strict patient-aligned fusion into six candidate configurations'
    )
    for organ in ['liver', 'spleen', 'muscle']:
        fuse.add_argument('--' + organ, required=True)
    fuse.add_argument('--clinical', required=True)
    fuse.add_argument('--clinical-features-json', required=True)
    fuse.add_argument('--out', required=True)
    backbone = sub.add_parser(
        'fine-tune',
        help='Two-stage Inception-v3 fine-tuning'
    )
    backbone.add_argument('--manifest', required=True)
    backbone.add_argument('--training-cohort', required=True)
    backbone.add_argument('--weights', required=True)
    backbone.add_argument('--organ', required=True, choices=['Liver', 'Spleen', 'Muscle'])
    backbone.add_argument('--device', default='cpu')
    backbone.add_argument('--seed', type=int, default=42)
    backbone.add_argument('--out', required=True)
    survival = sub.add_parser('survival', help='Endpoint-specific Cox and global log-rank tests')
    survival.add_argument('--training', required=True)
    survival.add_argument('--external', required=True)
    survival.add_argument('--out', required=True)
    cluster = sub.add_parser(
        'cluster',
        help='Training-only consensus clustering and 80:20 RF transfer'
    )
    cluster.add_argument('--training', required=True)
    cluster.add_argument('--external', required=True)
    cluster.add_argument('--out', required=True)
    organ = sub.add_parser(
        'organ-contributions',
        help='Training-only clustering sensitivity analyses'
    )
    organ.add_argument('--training', required=True)
    organ.add_argument('--out', required=True)
    extract = sub.add_parser(
        'extract',
        help='Inception-v3 features from prepacked five-slice image/mask inputs'
    )
    extract.add_argument('--manifest', required=True)
    extract.add_argument('--weights', required=True)
    extract.add_argument('--organ', required=True, choices=['Liver','Spleen','Muscle'])
    extract.add_argument('--device', default='cpu')
    extract.add_argument('--out', required=True)
    attr = sub.add_parser(
        'attribute',
        help='Permutation SHAP for the raw LSMFusion classifier'
    )
    attr.add_argument('--run', required=True)
    attr.add_argument('--cohort', choices=['internal','external'], default='external')
    attr.add_argument('--trust-model-files', action='store_true')
    attr.add_argument('--out', required=True)
    image_attr = sub.add_parser(
        'attribute-images',
        help='LSMFusion image-level ScoreCAM, occlusion and SHAP'
    )
    image_attr.add_argument('--config', required=True,
                           help='JSON with fusion_model, preprocessor, weights, images and background')
    image_attr.add_argument('--device', default='cpu')
    image_attr.add_argument('--trust-model-files', action='store_true')
    image_attr.add_argument('--out', required=True)
    select = sub.add_parser(
        'select-features',
        help='Training-only screening and 50% stability selection'
    )
    select.add_argument('--training', required=True)
    select.add_argument(
        '--internal',
        help='Apply training-fitted cleaner/scaler to this internal cohort'
    )
    select.add_argument('--external', help='Apply the same transform to this external cohort')
    select.add_argument(
        '--features-json',
        required=True,
        help='JSON array of candidate predictor names'
    )
    select.add_argument(
        '--frequency',
        type=float,
        default=.5,
        choices=[.5],
        help='Minimum selection frequency across 30 iterations (0.50)'
    )
    select.add_argument('--out', required=True)
    meld = sub.add_parser(
        'meld',
        help='Evaluate recorded MELD scores with bootstrap intervals'
    )
    meld.add_argument('--input', required=True)
    meld.add_argument('--out', required=True)
    profiles = sub.add_parser(
        'clinical-profiles',
        help='Clinical comparisons across fixed clusters'
    )
    profiles.add_argument('--input', required=True)
    profiles.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    if args.command == 'evaluate':
        from .evaluation import evaluate_runs
        result = evaluate_runs(args.runs_root, args.out, args.trust_model_files)
    elif args.command == 'evaluate-probabilities':
        from .evaluation import evaluate_fixed_file
        result = evaluate_fixed_file(args.input, args.out, args.threshold)
    elif args.command == 'train':
        from .training import fit_model
        result = fit_model(args.training_csv,args.model,args.out,args.benchmark,args.selection_dir)
    elif args.command == 'predict':
        from .prediction import predict_bundle
        result = predict_bundle(
            args.bundle,
            args.internal,
            args.external,
            args.out,
            args.trust_model_files
        )
    elif args.command == 'prepare-selected':
        from .preprocessing import prepare_selected
        result = prepare_selected(args.training, args.internal, args.external, args.model, args.out)
    elif args.command == 'fuse':
        import json
        from pathlib import Path
        from .fusion import run_fusion
        names = json.loads(Path(args.clinical_features_json).read_text(encoding='utf-8'))
        result = run_fusion(args.liver, args.spleen, args.muscle, args.clinical, names, args.out)
    elif args.command == 'fine-tune':
        from .backbone_training import fine_tune
        result = fine_tune(args.manifest, args.training_cohort, args.weights, args.organ,
                           args.out, args.device, args.seed)
    elif args.command == 'survival':
        from .survival import run_survival
        result = run_survival(args.training,args.external,args.out)
    elif args.command == 'cluster':
        from .clustering import run_clustering
        result = run_clustering(args.training,args.external,args.out)
    elif args.command == 'organ-contributions':
        from .organ_contributions import run_contributions
        result = run_contributions(args.training,args.out)
    elif args.command == 'extract':
        from .imaging import extract_features
        result = extract_features(args.manifest,args.weights,args.organ,args.out,args.device)
    elif args.command == 'attribute':
        from .attribution import run_attribution
        result = run_attribution(args.run,args.cohort,args.out,args.trust_model_files)
    elif args.command == 'attribute-images':
        from .imaging import attribute_images
        result = attribute_images(args.config,args.out,args.device,args.trust_model_files)
    elif args.command == 'select-features':
        import json
        from pathlib import Path
        from .selection import run_selection
        names=json.loads(Path(args.features_json).read_text(encoding='utf-8'))
        result=run_selection(
            args.training,
            names,
            args.frequency,
            args.out,
            args.internal,
            args.external
        )
    elif args.command == 'meld':
        from .clinical_benchmark import run_meld
        result=run_meld(args.input,args.out)
    elif args.command == 'clinical-profiles':
        from .clinical_profiles import run_profiles
        result=run_profiles(args.input,args.out)
    print(result.to_string(index=False) if hasattr(result,'to_string') else result)


if __name__ == '__main__':
    main()
