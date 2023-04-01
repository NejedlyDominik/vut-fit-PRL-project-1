/**
 * PRL - Project 1 - Parallel splitting
 *
 * login: xnejed09
 * name: Dominik Nejedly
 * year: 2023
 *
 * Main module containing main function performing parallel splitting algorithm
 */

#include <mpi.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib>

// predefined constants
#define SUCCESS 0
#define FAIL 1

#define ROOT_PROC 0
#define BYTE_SIZE 1
#define MIN_NUM_OF_IN_NUMS 8
#define INPUT_FILE "numbers"


/**
 * Join all subsequences split among all processes and print the resulting sequence with the specified output prefix.
 *
 * @param  resMsgPrefix  output prefix
 * @param  subSeqPtr     subsequence
 * @param  subSeqLen     length of subsequence
 * @param  size          number of processes
 * @param  rank          process number
 */
void joinSubSeqsAndPrintRes(std::string resMsgPrefix, uint8_t *subSeqPtr, int subSeqLen, int size, int rank) {
    int resSeqLen, shift;
    int *recvcounts = NULL;
    int *displs = NULL;

    if(rank == ROOT_PROC) {
        recvcounts = (int*) std::malloc(size * sizeof(int));
        displs = (int*) std::malloc(size * sizeof(int));
    }

    // Get the total number of elements in all subsequences.
    MPI_Reduce(&subSeqLen, &resSeqLen, 1, MPI_INT, MPI_SUM, ROOT_PROC, MPI_COMM_WORLD);
    // Get the numbers of elements in all subsequences.
    MPI_Gather(&subSeqLen, 1, MPI_INT, recvcounts, 1, MPI_INT, ROOT_PROC, MPI_COMM_WORLD);
    // Compute shift in the resulting sequence for every process.
    MPI_Exscan(&subSeqLen, &shift, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    // Get shifts in the resulting sequence of all processes.
    MPI_Gather(&shift, 1, MPI_INT, displs, 1, MPI_INT, ROOT_PROC, MPI_COMM_WORLD);

    uint8_t *resSeq = NULL;

    if(rank == ROOT_PROC) {
        // Set shift for process 0 to 0 (according to behaviour of MPI_Exscan it is undefined).
        displs[0] = 0;
        resSeq = (uint8_t*) std::malloc(resSeqLen);
    }

    // Get the resulting sequence.
    MPI_Gatherv(subSeqPtr, subSeqLen, MPI_UINT8_T, resSeq, recvcounts,
        displs, MPI_UINT8_T, ROOT_PROC, MPI_COMM_WORLD);

    // Print the resulting sequence with the specified output prefix.
    if(rank == ROOT_PROC) {
        std::cout << resMsgPrefix << "[";

        if(resSeqLen != 0) {
            std::cout << (unsigned) resSeq[0];

            for(int i = 1; i < resSeqLen; i++) {
                std::cout << ", " << (unsigned) resSeq[i];
            }
        }

        std::cout << "]\n";
    }

    if(rank == ROOT_PROC) {
        std::free(recvcounts);
        std::free(displs);
        std::free(resSeq);
    }
}


int main(int argc, char **argv) {
    int rank, size, subSeqNumsLen;
    uint8_t pseudoMedian;
    std::vector<uint8_t> numbers;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Load numbers from input file and get pseudo median (middle element of input sequence) and subsequence length.
    if(rank == ROOT_PROC) {
        std::ifstream inFile(INPUT_FILE, std::ios::binary);

        if(!inFile.is_open()) {
            std::cerr << "Input file " << INPUT_FILE << " cannot be opened.\n";
            MPI_Abort(MPI_COMM_WORLD, FAIL);
        }

        numbers.reserve(MIN_NUM_OF_IN_NUMS);        
        uint8_t inNum;

        while(inFile.read((char*) &inNum, BYTE_SIZE)) {
            numbers.push_back(inNum);
        }

        inFile.close();
        int numbersLen = numbers.size();

        if(numbersLen <= 0) {
            std::cerr << "Empty input sequence of numbers.\n";
            MPI_Abort(MPI_COMM_WORLD, FAIL);
        }

        MPI_Comm_size(MPI_COMM_WORLD, &size);
        
        if (numbersLen % size != 0) {
            std::cerr << "The number of elements (" << numbersLen << ") in the input sequence of numbers "
                "is not divisible without remainder by the number of processes (" << size << ").\n";
            MPI_Abort(MPI_COMM_WORLD, FAIL);
        }

        subSeqNumsLen = numbersLen / size;
        int pseudoMedianIdx = (numbersLen % 2 == 0) ? (numbersLen / 2) - 1 : numbersLen / 2;
        pseudoMedian = numbers.at(pseudoMedianIdx);
    }

    // Broadcast subsequence length and selected pseudo median to all processes.
    MPI_Bcast(&subSeqNumsLen, 1, MPI_INT, ROOT_PROC, MPI_COMM_WORLD);
    MPI_Bcast(&pseudoMedian, 1, MPI_UINT8_T, ROOT_PROC, MPI_COMM_WORLD);

    uint8_t *subSeqNumbers =(uint8_t*) std::malloc(subSeqNumsLen);

    // Split the input sequence among all processes.
    MPI_Scatter(numbers.data(), subSeqNumsLen, MPI_UINT8_T, subSeqNumbers,
        subSeqNumsLen, MPI_UINT8_T, ROOT_PROC, MPI_COMM_WORLD);

    std::vector<uint8_t> subSeqLower;
    std::vector<uint8_t> subSeqEqual;
    std::vector<uint8_t> subSeqGreater;

    // Sort the assigned subsequence by the pseudo median.
    for(int i = 0; i < subSeqNumsLen; i++) {
        uint8_t num = subSeqNumbers[i];

        if(num < pseudoMedian) {
            subSeqLower.push_back(num);
        }
        else if(num > pseudoMedian) {
            subSeqGreater.push_back(num);
        }
        else {
            subSeqEqual.push_back(num);
        }
    }

    // Print the resulting distribution.
    joinSubSeqsAndPrintRes("L: ", subSeqLower.data(), subSeqLower.size(), size, rank);
    joinSubSeqsAndPrintRes("E: ", subSeqEqual.data(), subSeqEqual.size(), size, rank);
    joinSubSeqsAndPrintRes("G: ", subSeqGreater.data(), subSeqGreater.size(), size, rank);

    if(rank == ROOT_PROC) {
        std::free(subSeqNumbers);
    }

    MPI_Finalize();
    return SUCCESS;
}
