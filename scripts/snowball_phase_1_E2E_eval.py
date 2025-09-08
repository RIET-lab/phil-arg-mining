import rootutils
rootutils.setup_root(__file__, indicator=".git")
from pathlib import Path
from moralkg import Config, get_logger
from moralkg.argmining.loaders import Dataset
from moralkg.snowball.phase_1.evals import End2EndEvaluator

def main():    
    # Setup
    Config.load()
    logger = get_logger("snowball_evaluator")
    
    # Load dataset
    logger.info("Loading dataset...")
    dataset = Dataset()

    # Create evaluator
    parsed_argmap_dir = Path("datasets/processed/argument_mining/snowball_phase_1/end2end/standard")
    evaluator = End2EndEvaluator(dataset, parsed_argmap_dir=parsed_argmap_dir, save_parsed_argument_json=True, use_existing_parsed_if_found=True)

    output_file = Path("datasets/interim/argument_mining/snowball_phase_1/end2end/standard/generation_outputs_standard_few-shot_one-shot_zero-shot_default_zs1_zs2_20250901_132248.json")
    output_dir = output_file.parent
    results = evaluator.evaluate_output_files([output_file])


    # Print summary
    logger.info("="*50)
    logger.info("EVALUATION RESULTS SUMMARY")
    logger.info("="*50)
    logger.info(f"Total papers: {results.total_papers}")
    logger.info(f"Successful parses: {results.successful_parses}")
    logger.info(f"Failed parses: {results.failed_parses}")
    logger.info(f"Parse success rate: {results.parse_success_rate:.2%}")
    
    if results.aggregate_metrics:
        logger.info("\nAggregate Metrics:")
        for metric, value in results.aggregate_metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
    
    if results.strategy_breakdown:
        logger.info("\nStrategy Breakdown:")
        for strategy, metrics in results.strategy_breakdown.items():
            logger.info(f"  {strategy}:")
            for metric, value in metrics.items():
                logger.info(f"    {metric}: {value:.4f}")
    
    # Save results
    eval_output_file = output_dir / "evaluation_results.json"
    evaluator.save_results(results, eval_output_file)
    

if __name__ == "__main__":
    main()