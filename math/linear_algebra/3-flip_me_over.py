#!/usr/bin/env python3

def matrix_transpose(matrix):
    transpose = []
    for col in range(len(matrix[0])):
        temp_matrix = []
        for row in range(len(matrix)):
            temp_matrix.append(matrix[row][col])
        transpose.append(temp_matrix)
    return transpose
