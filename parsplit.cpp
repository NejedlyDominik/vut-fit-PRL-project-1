/**
 * PRL - Project 1 - Parallel splitting
 *
 * login: xnejed09
 * name: Dominik Nejedly
 * year: 2023
 *
 * Main module containing main function performing parallel splitting algorithm
 **/

#include <mpi.h>
#include <iostream>
#include <fstream>
#include <vector>

#define SUCCESS 0
#define FAIL 1

#define ROOT_PROC 0
#define BYTE_SIZE 1
#define MIN_NUM_OF_IN_NUMS 8
#define INPUT_FILE "numbers"


void concatSubSeqsAndPrintRes(std::string resMsgPrefix, uint8_t *subSeqPtr, int subSeqLen, int size, int rank) {
    int seqLen, shift;
    std::vector<int> recvcounts;
    std::vector<int> displs;

    if(rank == ROOT_PROC) {
        recvcounts.resize(size);
        displs.resize(size);
    }

    MPI_Reduce(&subSeqLen, &seqLen, 1, MPI_INT, MPI_SUM, ROOT_PROC, MPI_COMM_WORLD);
    MPI_Gather(&subSeqLen, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, ROOT_PROC, MPI_COMM_WORLD);
    MPI_Exscan(&subSeqLen, &shift, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Gather(&shift, 1, MPI_INT, displs.data(), 1, MPI_INT, ROOT_PROC, MPI_COMM_WORLD);

    std::vector<uint8_t> resSeq;

    if(rank == ROOT_PROC) {
        displs.at(0) = 0;
        resSeq.resize(seqLen);
    }

    MPI_Gatherv(subSeqPtr, subSeqLen, MPI_UINT8_T, resSeq.data(), recvcounts.data(),
        displs.data(), MPI_UINT8_T, ROOT_PROC, MPI_COMM_WORLD);

    if(rank == ROOT_PROC) {
        std::cout << resMsgPrefix << "{";

        if(!resSeq.empty()) {
            std::cout << (unsigned) resSeq.front();

            for(std::vector<uint8_t>::iterator numIt = ++resSeq.begin(); numIt != resSeq.end(); ++numIt) {
                std::cout << ", " << (unsigned) *numIt;
            }
        }

        std::cout << "}\n";
    }
}


int main(int argc, char **argv) {
    int rank, size, subSeqNumsLen;
    uint8_t pseudoMedian;
    std::vector<uint8_t> numbers;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if(rank == ROOT_PROC) {
        std::ifstream inFile(INPUT_FILE, std::ios::binary);

        MPI_Comm_size(MPI_COMM_WORLD, &size);

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
        
        if (numbersLen % size != 0) {
            std::cerr << "The number of elements in the input sequence of numbers "
                "is not divisible without remainder by the number of processes.\n";
            MPI_Abort(MPI_COMM_WORLD, FAIL);
        }

        subSeqNumsLen = numbersLen / size;
        int pseudoMedianIdx = (numbersLen % 2 == 0) ? (numbersLen / 2) - 1 : numbersLen / 2;
        pseudoMedian = numbers.at(pseudoMedianIdx);
    }

    MPI_Bcast(&subSeqNumsLen, 1, MPI_INT, ROOT_PROC, MPI_COMM_WORLD);
    MPI_Bcast(&pseudoMedian, 1, MPI_UINT8_T, ROOT_PROC, MPI_COMM_WORLD);

    std::vector<uint8_t> subSeqNumbers(subSeqNumsLen);

    MPI_Scatter(numbers.data(), subSeqNumsLen, MPI_UINT8_T, subSeqNumbers.data(),
        subSeqNumsLen, MPI_UINT8_T, ROOT_PROC, MPI_COMM_WORLD);

    std::vector<uint8_t> subSeqLower;
    std::vector<uint8_t> subSeqEqual;
    std::vector<uint8_t> subSeqGreater;

    for(uint8_t num: subSeqNumbers) {
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

    concatSubSeqsAndPrintRes("L: ", subSeqLower.data(), subSeqLower.size(), size, rank);
    concatSubSeqsAndPrintRes("E: ", subSeqEqual.data(), subSeqEqual.size(), size, rank);
    concatSubSeqsAndPrintRes("G: ", subSeqGreater.data(), subSeqGreater.size(), size, rank);

    MPI_Finalize();
    return SUCCESS;
}
