from scripts.run_part1 import main as run_part1
from scripts.run_part2 import main as run_part2
from scripts.run_part3 import main as run_part3
from scripts.run_part4_sarimax_temp import main as run_part4
from scripts.run_part5_tree_weekly import main as run_part5
from scripts.run_part6_plot_lstm import main as run_part6_plot
from scripts.make_model_comparison import main as run_model_comparison_csv
from scripts.run_model_comparison_plot import main as run_model_comparison_plot


def main():
    print("\n=== Running Part 1 ===")
    run_part1()

    print("\n=== Running Part 2 ===")
    run_part2()

    print("\n=== Running Part 3 ===")
    run_part3()

    print("\n=== Running Part 4 ===")
    run_part4()

    print("\n=== Running Part 5 ===")
    run_part5()

    print("\n=== Running Part 6 plot ===")
    run_part6_plot()

    print("\n=== Building model comparison CSV ===")
    run_model_comparison_csv()

    print("\n=== Building model comparison plot ===")
    run_model_comparison_plot()

    print("\n=== Pipeline complete ===")


if __name__ == "__main__":
    main()