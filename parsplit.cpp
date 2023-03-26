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

int main(int argc, char **argv) {
    int rank, subSeqLen;
    uint8_t pseudoMedian;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if(rank == ROOT_PROC) {
        int size;
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        std::ifstream inFile(INPUT_FILE, std::ios::binary);

        if(!inFile.is_open()) {
            std::cerr << "Input file " << INPUT_FILE << " cannot be opened.\n";
            MPI_Abort(MPI_COMM_WORLD, FAIL);
        }

        std::vector<uint8_t> numbers;
        numbers.reserve(MIN_NUM_OF_IN_NUMS);
        
        uint8_t num;

        while(inFile.read((char*) &num, BYTE_SIZE)) {
            numbers.push_back(num);
        }

        inFile.close();
        int numbersLen = numbers.size();

        if(numbersLen <= 0) {
            std::cerr << "Empty input sequence of numbers.\n";
            MPI_Abort(MPI_COMM_WORLD, FAIL);
        }
        else if (numbersLen % size != 0) {
            std::cerr << "The number of elements in the input sequence of numbers "
                "is not divisible without remainder by the number of processes.\n";
            MPI_Abort(MPI_COMM_WORLD, FAIL);
        }

        subSeqLen = numbersLen / size;
        MPI_Bcast(&subSeqLen, 1, MPI_INT, ROOT_PROC, MPI_COMM_WORLD);

        int pseudoMedianIdx = numbersLen / 2;

        if(numbersLen % 2 == 0) {
            pseudoMedianIdx--;
        }

        pseudoMedian = numbers.at(pseudoMedianIdx);
        MPI_Bcast(&pseudoMedian, 1, MPI_UINT8_T, ROOT_PROC, MPI_COMM_WORLD);
    }
    else {
        MPI_Bcast(&subSeqLen, 1, MPI_INT, ROOT_PROC, MPI_COMM_WORLD);
        MPI_Bcast(&pseudoMedian, 1, MPI_UINT8_T, ROOT_PROC, MPI_COMM_WORLD);
    }

    //TODO

    MPI_Finalize();
    return SUCCESS;
}