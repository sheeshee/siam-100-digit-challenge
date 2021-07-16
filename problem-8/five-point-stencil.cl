__kernel void matvec(
    __global double *vector,
    __global double *result,
    double alpha,
    double beta,
    int XR,
    int XC
)
{
// Get Global Identifiers i.e. the row or col index
int row_index = get_global_id(0);
int col_index = get_global_id(1);

// Determine boolean operators.
// These decide if a particular term is active or not.
int b1 = (int) (row_index > 0);      // bottom
int b2 = (int) (row_index < XR - 1); // top
int b3 = (int) (col_index > 0);      // left
int b4 = (int) (col_index < XC - 1); // right

// Location of center of the stencil
int loc = row_index*XC + col_index;

// Apply the stencil
double product = alpha*vector[loc] // center
    + b1*beta*vector[b1*(loc-XC)]  // bottom
    + b2*beta*vector[b2*(loc+XC)]  // top
    + b3*beta*vector[b3*(loc-1)]   // left
    + b4*beta*vector[b4*(loc+1)];  // right

// Output answer
result[loc] = product;
}
