#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <chrono>
#include <functional>
#include <mpi.h>

const int ARRAY_SIZE = 100000;

void fill_random_vector(std::vector<int>& vec, int size, int min_value, int max_value) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> distribution(min_value, max_value);

    vec.resize(size);
    for (int i = 0; i < size; i++) {
        vec[i] = distribution(gen);
    }
}

void insertion_sort(std::vector<int>& vec, int start, int end) {
    for (int i = start + 1; i <= end; i++) {
        int key = vec[i];
        int j = i - 1;
        while (j >= start && vec[j] > key) {
            vec[j + 1] = vec[j];
            j--;
        }
        vec[j + 1] = key;
    }
}

std::vector<int> merge_sorted_vectors(const std::vector<int>& vec1, const std::vector<int>& vec2) {
    std::vector<int> merged_vec;
    merged_vec.reserve(vec1.size() + vec2.size());
    std::merge(vec1.begin(), vec1.end(), vec2.begin(), vec2.end(), std::back_inserter(merged_vec));
    return merged_vec;
}

std::vector<int> parallel_sort(const std::vector<int>& input_vec, int world_rank, int world_size) {
    int size = input_vec.size();

    int chunk_size = size / world_size;
    int start = world_rank * chunk_size;
    int end = (world_rank == world_size - 1) ? size - 1 : start + chunk_size - 1;

    std::vector<int> sorted_segment(input_vec.begin() + start, input_vec.begin() + end + 1);
    insertion_sort(sorted_segment, 0, end - start);

    std::vector<int> sorted_vec_segments(size);
    MPI_Gather(sorted_segment.data(), chunk_size, MPI_INT, sorted_vec_segments.data(), chunk_size, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<int> final;
    final.reserve(size);

    if (world_rank == 0) {
        for (int i = 0; i < world_size; ++i) {
            int seg_start = i * chunk_size;
            int seg_end = (i == world_size - 1) ? size - 1 : seg_start + chunk_size - 1;
            std::vector<int> seg_vec(sorted_vec_segments.begin() + seg_start, sorted_vec_segments.begin() + seg_end + 1);
            final = merge_sorted_vectors(final, seg_vec);
        }
    }

    return final;
}

std::vector<int> non_parallel_sort(const std::vector<int>& input_vec, int world_rank, int world_size) {
    int size = input_vec.size();
    std::vector<std::vector<int>> sorted_segments(world_size);

    for (int i = 0; i < world_size; ++i)
    {
        int chunk_size = size / world_size;
        int start = i * chunk_size;
        int end = (i == world_size - 1) ? size - 1 : start + chunk_size - 1;

        std::vector<int> sorted_segment(input_vec.begin() + start, input_vec.begin() + end + 1);
        insertion_sort(sorted_segment, 0, end - start);

        sorted_segments[i] = std::move(sorted_segment);
    }
    std::vector<int> final_sorted_vec = sorted_segments[0];
    for (int i = 1; i < world_size; i++) {
        final_sorted_vec = merge_sorted_vectors(final_sorted_vec, sorted_segments[i]);
    }

    return final_sorted_vec;
}

void sort_using(std::function<std::vector<int>(const std::vector<int>&, int, int)> sort_func, const std::vector<int>& input_vec, int world_rank, int world_size)
{
    auto start_time = std::chrono::steady_clock::now();
    std::vector<int> sorted_array = sort_func(input_vec, world_rank, world_size);
    auto end_time = std::chrono::steady_clock::now();

    if (world_rank == 0) {
        std::cout << "Sorted array (first 10 elements):" << std::endl;
        for (int i = 0; i < 10; i++) {
            std::cout << sorted_array[i] << " ";
        }
        std::cout << std::endl;

        std::cout << "Execution time: " << std::chrono::duration<double>(end_time - start_time).count() << " seconds" << std::endl;
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    std::vector<int> input_array(ARRAY_SIZE);

    if (world_rank == 0) {
        fill_random_vector(input_array, ARRAY_SIZE, 1, 10000);

        std::cout << "Generated array (first 10 elements):" << std::endl;
        for (int i = 0; i < 10; i++) {
            std::cout << input_array[i] << " ";
        }
        std::cout << std::endl;
    }

    MPI_Bcast(input_array.data(), ARRAY_SIZE, MPI_INT, 0, MPI_COMM_WORLD);

    if (world_rank == 0)
        std::cout << "Non-parallel version:" << std::endl;
    sort_using(non_parallel_sort, input_array, world_rank, world_size);

    if (world_rank == 0)
        std::cout << "Parallel version:" << std::endl;
    sort_using(parallel_sort, input_array, world_rank, world_size);

    MPI_Finalize();

    return 0;
}
