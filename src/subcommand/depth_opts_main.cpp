#include "subcommand.hpp"
#include "odgi.hpp"
#include "position.hpp"
#include "args.hxx"
#include "split.hpp"
#include "algorithms/bfs.hpp"
#include "algorithms/depth.hpp"
#include "algorithms/path_length.hpp"
#include <omp.h>

#include "src/algorithms/subgraph/extract.hpp"

namespace odgi {

    using namespace odgi::subcommand;

    int main_depth_opt(int argc, char **argv) {

        // trick argumentparser to do the right thing with the subcommand
        for (uint64_t i = 1; i < argc - 1; ++i) {
            argv[i] = argv[i + 1];
        }
        std::string prog_name = "odgi depth";
        argv[0] = (char *) prog_name.c_str();
        --argc;

        args::ArgumentParser parser("Find the depth of a graph as defined by query criteria. Without specifying any non-mandatory options, it prints in a tab-delimited format path, start, end, and mean.depth to stdout.");
        args::Group mandatory_opts(parser, "[ MANDATORY OPTIONS ]");
        args::ValueFlag<std::string> og_file(mandatory_opts, "FILE", "Load the succinct variation graph in ODGI format from this *FILE*. The file name usually ends with *.og*. It also accepts GFAv1, but the on-the-fly conversion to the ODGI format requires additional time!", {'i', "input"});
        args::Group depth_opts(parser, "[ Depth Options ]");
        args::ValueFlag<std::string> _subset_paths(depth_opts, "FILE",
                                                  "Compute the depth considering only the paths specified in the FILE. "
                                                  "The file must contain one path name per line and a subset of all paths can be specified; "
                                                  "If a step is of a path of the given list, it is taken into account when calculating a node's depth. Else not.",
                                                  {'s', "subset-paths"});
        args::ValueFlag<uint64_t> random(depth_opts, "N",
                          "Print a randomly generated, newline-separated subset of paths specified in the FILE. "
                          "The percentage of paths selected is determined by N. Duplicates are permitted.",
                          {'r', "random-subset"});
        args::Flag graph_depth_vec(depth_opts, "graph-depth-vec",
                                   "Compute the depth on each node in the graph, writing a vector by base in one line.",
                                   {'v', "graph-depth-vec"}, "boolean");
        args::ValueFlag<std::string> _optimizations(depth_opts, "optimizations",
                                                    "The optimization function to be benchmarked.",
                                                    {'O', "depth-opt"}, "baseline");
        // args::Flag<std::string> opt_subset_paths(depth_opts, "subset_paths_optimization",
        //                                                    "Flag to determine whether or not subset_paths is computed in parallel.",
        //                                                    {'S', "opt-subset-paths"});

        args::Group threading_opts(parser, "[ Threading ] ");
        args::ValueFlag<uint64_t> _num_threads(threading_opts, "N", "Number of threads to use in parallel operations.", {'t', "threads"});
		args::Group processing_info_opts(parser, "[ Processing Information ]");
		args::Flag progress(processing_info_opts, "progress", "Write the current progress to stderr.", {'P', "progress"});
        args::Group program_info_opts(parser, "[ Program Information ]");
        args::HelpFlag help(program_info_opts, "help", "Print a help message for odgi depth.", {'h', "help"});
        try {
            parser.ParseCLI(argc, argv);
        } catch (args::Help) {
            std::cout << parser;
            return 0;
        } catch (args::ParseError e) {
            std::cerr << e.what() << std::endl;
            std::cerr << parser;
            return 1;
        }
        if (argc == 1) {
            std::cout << parser;
            return 1;
        }

        if (!og_file) {
            std::cerr << "[odgi::depth-opts] error: please specify a target graph via -i=[FILE], --idx=[FILE]." << std::endl;
            return 1;
        }

		const uint64_t num_threads = args::get(_num_threads) ? args::get(_num_threads) : 1;

		odgi::graph_t graph;
        assert(argc > 0);
        if (!args::get(og_file).empty()) {
            const std::string infile = args::get(og_file);
            if (infile == "-") {
                graph.deserialize(std::cin);
            } else {
				utils::handle_gfa_odgi_input(infile, "depth", args::get(progress), num_threads, graph);
            }
        }

        omp_set_num_threads((int) num_threads);
		const uint64_t shift = graph.min_node_id();
        if (graph.max_node_id() - shift >= graph.get_node_count()){
            std::cerr << "[odgi::depth-opts] error: the node IDs are not compacted. Please run 'odgi sort' using -O, --optimize to optimize the graph." << std::endl;
            exit(1);
        }

        // time how long it takes to execute
        auto t1 = std::chrono::high_resolution_clock::now();

        // Compute paths_to_consider
        const std::string optimizations = args::get(_optimizations);
        std::vector<bool> paths_to_consider;
        if (_subset_paths) {
            paths_to_consider.resize(graph.get_path_count() + 1, false);

            std::ifstream refs(args::get(_subset_paths).c_str());
            std::string line;

            while (std::getline(refs, line)) {
                if (!line.empty()) {
                    if (!graph.has_path(line)) {
                        std::cerr << "[odgi::depth-opts] error: path " << line << " not found in graph" << std::endl;
                        exit(1);
                    }
                    paths_to_consider[as_integer(graph.get_path_handle(line))] = true;
                }
            }
        } else {
            paths_to_consider.resize(graph.get_path_count() + 1, true);
        }

        auto t2 = std::chrono::high_resolution_clock::now();

        // Compute depths
        auto get_graph_node_depth = [](const odgi::graph_t &graph, const handle_t h,
                                       const std::vector<bool>& paths_to_consider) {
            uint64_t node_depth = 0;

            graph.for_each_step_on_handle(
                h,
                [&](const step_handle_t &step_h) {
                    if (paths_to_consider[
                            as_integer(graph.get_path_handle_of_step(step_h))]) {
                        ++node_depth;
                    }
                });

            return node_depth;
        };

        std::vector<uint64_t> depths(graph.get_node_count() + 1);
        if (optimizations == "baseline") {
            // For optional ablation 4, could package into tasks and use a task-driven workflow
            graph.for_each_handle(
                [&](const handle_t& h) {
                    auto id = graph.get_id(h);
                    depths[id - shift] = get_graph_node_depth(graph, h, paths_to_consider);
                }, false);
        } else {
            graph.for_each_handle(
                [&](const handle_t& h) {
                    auto id = graph.get_id(h);
                    depths[id - shift] = get_graph_node_depth(graph, h, paths_to_consider);
                }, true);
        }

        auto t3 = std::chrono::high_resolution_clock::now();

        // Print output to commandline if requested
        if (graph_depth_vec) {
            std::cout << (og_file ? args::get(og_file) : "graph") << "_vec";
            // int max_depth = 0;
            for (uint64_t i = 0; i < graph.get_node_count(); ++i) {
                std::cout << " " << depths[i];
                // if (depths[i] > max_depth) {
                //     max_depth = depths[i];
                // }
            }
            // std::cout << "max depth: " << max_depth;
            std::cout << std::endl;
        } else if (random) { // If specified, print a random select of the paths in our input graph
            const uint64_t p = graph.get_path_count();
            const uint64_t n = args::get(random) * p/100;
            std::vector<path_handle_t> path_handles(p);
            int i = 0;
            graph.for_each_path_handle([&](const path_handle_t &path_h) {
                path_handles[i] = path_h;
                i++;
            });
            for (uint64_t i = 0; i < n; i++) {
                std::cout << graph.get_path_name(path_handles[std::rand() % p]) << std::endl;
            }
        } else { // Print how long it takes the node depth computation to execute. subset\tdepth\ttotal
            // Get time in milliseconds as a double
            std::chrono::duration<double, std::milli> subset_time = t2 - t1;
            std::chrono::duration<double, std::milli> depths_time = t3 - t2;
            std::chrono::duration<double, std::milli> total_time = t3 - t1;
            std::cout << subset_time.count() << "\t" << depths_time.count() << "\t" << total_time.count() << std::endl;
        }

        return 0;
    }

    static Subcommand odgi_depth_opt("depth-opts", "Compute the validity and runtime of optimized versions of node depth.",
                                 PIPELINE, 3, main_depth_opt);

}
