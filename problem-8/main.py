"""
This script solves Problem 8 of the SIAM 100 digit challenge, printing
the solution to stdout. It will simulate the system at each timestep until
the center reaches the targetted value.

Use as `> python main.py`

See the section at the end for the main code block for more editting parameters
and more information.
"""
import logging
import os
import time
from pathlib import Path
from typing import NamedTuple

import attr
import matplotlib.pyplot as plt
import numpy as np
import pyopencl as cl
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import LinearOperator, cg, spilu


class Timer:
    """
    A simple timer class to measure the execution time
    of a function (seconds) in a ```with``` block.
    Taken from module lecture notes.
    """
    def __enter__(self):
        """
        Measure time at start.
        """
        self.start = time.time()
        return self

    def __exit__(self, *args):
        """
        Measure time at end and take difference
        from start.
        """
        self.end = time.time()
        self.interval = self.end - self.start


@attr.s
class Solver8:
    """
    A class to contain all the parameters of the problem and implement the
    different methods needed to yield a solution.
    """
    target_spacial_step: float = attr.ib()
    time_step:float = attr.ib()
    bottom_wall_bound = attr.ib(default=5)
    system_size = attr.ib(init=False)
    spacial_step = attr.ib(init=False)
    rhs_operator = attr.ib(init=False)
    ctx = attr.ib(init=False)

    def __attrs_post_init__(self):
        """
        Calculate the system size from the spacial step.
        """
        self.ctx = cl.create_some_context()
        self.system_size = _calculate_system_size(self.target_spacial_step)
        self.spacial_step = _calculate_spacial_step(self.system_size)
        self.rhs_operator = FivePointStencil(self, 1 - 4*self.alpha, self.alpha)
        self.lhs_operator = FivePointStencil(self, 1 + 4*self.alpha, -self.alpha)
        self.boundary_addition = self.define_boundary_vector()

        self.lhs_inverted_operator = InvertedOperator(self.lhs_operator)

    @property
    def array_length(self):
        """
        Return the length of the system when it is represented as a flattened
        1-D array.f
        """
        return self.rows * self.columns

    @property
    def rows(self):
        """
        Return the number of rows of the system.
        """
        return self.system_size

    @property
    def columns(self):
        """
        Return the number of columns of the system.
        """
        return self.system_size

    @property
    def alpha(self):
        """
        Calculate alpha term - see equations in introduction
        """
        # Set stepping term
        alpha = self.time_step/self.spacial_step**2/2
        return alpha

    def get_initial_state(self):
        """
        Return an array representing the system in its initial state.
        """
        u = np.zeros(self.array_length)
        return u

    def define_boundary_vector(self):
        """
        Translates the influence of the boundary conditions into a vector
        that can be applied in the numerical scheme.
        """
        b = np.zeros(self.array_length)
        k = 0
        for row_idx in range(self.rows):
            for _ in range(self.columns):
                # BOTTOM
                if row_idx == 0:
                    b[k] += self.bottom_wall_bound
                k += 1

        return 2*b*self.alpha

    def show(self, system, title=''):
        """
        Display a colour plot image of the solution on the square.

        Boundary conditions must be included in the passed solution.

        Parameters
        ----------
        system : np.array
            the flattened solution to plot

        Returns nothing
        """
        p = system.reshape(self.rows, self.columns)
        # Display the solution
        plt.imshow(p, origin='lower')
        plt.colorbar()
        plt.title(title)
        plt.show()

    def advance_system(self, system):
        """
        Evolve the system to its new state after a discrete timestep of size
        tau.
        """
        rhs = self.rhs_operator @ system + self.boundary_addition
        new_system, _ = self.lhs_inverted_operator @ rhs
        return new_system

    def simulate_to(self, target_time: float, start_time=0, start_state=None):
        """
        Simulate the system up to the time specified.
        """
        elapsed_time = start_time
        if start_state is None:
            temperature_profile = self.get_initial_state()
        else:
            temperature_profile = start_state
        while elapsed_time < target_time:
            temperature_profile = self.advance_system(temperature_profile)
            elapsed_time += self.time_step
        return temperature_profile, elapsed_time

    def stop_when_center_is(self, target_temperature):
        """
        Return the pair of values that surround the target center temperature.
        """
        u = self.get_initial_state()
        center = self.get_center_temperature(u)
        elapsed_time = 0
        while center < target_temperature:
            with Timer() as t:
                prev = (center, elapsed_time)
                u = self.advance_system(u)
                elapsed_time += self.time_step
                center = self.get_center_temperature(u)
            print("T %5.4f - %.4f s" % (elapsed_time, t.interval))
        overstep = (center, elapsed_time)
        return prev, overstep, u

    def get_center_temperature(self, temperature_profile):
        """
        Retrieve the temperature at the center of the plate
        """
        plate = temperature_profile.reshape(self.rows, self.columns)
        if self.rows % 2 == 0:
            row_points = [self.rows//2-1, self.rows//2]
        else:
            row_points = [self.rows//2, self.rows//2]
        if self.columns % 2 == 0:
            col_points = [self.columns//2-1, self.columns//2]
        else:
            col_points = [self.columns//2, self.columns//2]
        points = plate[row_points, col_points].ravel()
        mean_value = sum(points)/len(points)
        return mean_value

    def center_temperature_at(self, target_time, start_time=0, start_state=None):
        """
        Return the temperature at the center of the plate at a given time, along
        with how much time has elapsed up to that point.
        """
        plate, elapsed_time = self.simulate_to(target_time, start_time, start_state)
        return self.get_center_temperature(plate), plate, elapsed_time


@attr.s
class FivePointStencil(LinearOperator):
    """
    An SciPy Linear Operator that used an OpenCL kernel to apply the
    spatial five point stencil to the system
    """
    solver: Solver8 = attr.ib()
    alpha: float = attr.ib()
    beta: float = attr.ib()
    drop_tol: float = attr.ib(default=1e-5)
    fill_factor: float = attr.ib(default=18)
    shape = attr.ib(init=False)
    queue = attr.ib(init=False)
    kernel = attr.ib(init=False)
    matvec = attr.ib(init=False)
    dtype = np.dtype(np.float64)

    def __attrs_post_init__(self):
        """
        Setup the OpenCL environment and kernel
        """
        self.shape = (self.solver.array_length, self.solver.array_length)
        self.queue, self.kernel = self._setup_cl_objects()
        self.matvec = self.get_matvec()

    def import_opencl_script(self, filename="five-point-stencil.cl"):
        """
        Reads the given file.
        """
        dir = Path(__file__).parent
        with open(dir/filename) as f:
            contents = f.read()
        return contents

    def _setup_cl_objects(self):
        """
        Creates the OpenCL context and queue objects
        """
        # queue
        queue = cl.CommandQueue(self.solver.ctx)
        # Build the Kernel
        prg = cl.Program(self.solver.ctx, self.import_opencl_script())
        prg.build()
        kernel = prg.matvec
        return queue, kernel

    def get_matvec(self):
        """
        Setup the OpenCL kernel using the USE_MAPPED_BUFFER flag to choose which
        type of buffer to create.
        """
        if USE_MAPPED_BUFFER:
            BufferChoice = MappedBuffer
        else:
            BufferChoice = Buffer
        # use buffer choice to generate two buffer objects.
        input_buffer = BufferChoice(
            self.solver.ctx, self.queue,
            self.solver.array_length, np.double,
            pgr_read=True
        )
        output_buffer = BufferChoice(
            self.solver.ctx, self.queue,
            self.solver.array_length, np.double,
            pgr_write=True
        )
        # now define a matvec function within this context
        def matvec(x):
            """
            An OpenCL enabled matvec operator applying the FivePointStencil
            """
            input_buffer.write(x)
            # Run the kernel - Refer to call method below
            # https://documen.tician.de/pyopencl/runtime_program.html#pyopencl.Kernel.__call__
            self.kernel(
                # The OpenCL work item queue
                self.queue,
                # global size - overall size of the computational grid:
                # one work item will be launched for every integer point
                # in the grid
                (self.solver.rows, self.solver.columns),
                # local size - can be set to None, in in which case the
                # implementation will use an implementation-defined
                # workgroup size
                None,
                # Buffers
                input_buffer.as_cl(), output_buffer.as_cl(),
                # Other parameters passed to kernel
                np.float64(self.alpha), np.float64(self.beta),
                np.int32(self.solver.rows), np.int32(self.solver.columns)
            )
            self.queue.finish()
            return output_buffer.read()
        return matvec

    def bytesize(self, dtype_str):
        """
        Return the size in bytes of the type given in string form
        """
        return memory_size(self.solver.array_length, dtype_str)

    def _matvec(self, x):
        """
        Apply the initialised operator to the given vector.
        """
        return self.matvec(x)

    def _build_sparse_matrix(self):
        """
        Build a matrix which applies the Crank-Nicholson finite difference
        scheme to solve the problem. This is used only to create the preconditioner.
        The operator for applying this is an OpenCl kernel.
        """
        data = []
        rows = []
        cols = []

        def add(datasource, val, row, colshift):
            """
            Add coefficient to operator.
            """
            datasource.append(val)
            rows.append(row)
            if row+colshift < 0:
                msg = f'Negative col index {row}: {colshift}'
                raise Exception(msg)
            cols.append(row+colshift)

        k = 0
        for row_idx in range(0, self.solver.rows):
            for col_idx in range(0, self.solver.columns):
                # center
                add(data, self.alpha, k, 0)

                # left
                if col_idx >= 1:
                    add(data, self.beta, k, -1)

                # right
                if col_idx < self.solver.columns - 1:
                    add(data, self.beta, k, 1)

                # top
                if row_idx < self.solver.rows - 1:
                    add(data, self.beta, k, self.solver.columns)

                # bottom
                if row_idx >= 1:
                    add(data, self.beta, k, -self.solver.columns)

                k += 1

        # Check for negative column indexes
        if any([x<0 for x in cols]):
            msg = f'Negative column index {k}'
            raise Exception(msg)

        A = coo_matrix((data, (rows, cols))).tocsc()

        # Ensure matrix is square
        if A.shape[0] != A.shape[1]:
            msg = f'Matrix is not square: {A.shape}'
            raise Exception(msg)

        # Ensure it's the expected size
        if A.shape[0] != self.solver.array_length:
            msg = f'Matrix wrong size:{A.shape[0]}'
            raise Exception(msg)

        return A

    def get_preconditioner(self):
        """
        Using SPILU and a specified drop tolerance, create a preconditioner
        for A.
        """
        A = self._build_sparse_matrix()
        ilu = spilu(A, drop_tol=self.drop_tol, fill_factor=self.fill_factor)
        Mx = lambda x: ilu.solve(x.astype(np.float32))
        N = A.shape[0]
        P = LinearOperator((N, N), Mx)
        return P


@attr.s
class Buffer:
    """
    Wrapper around the pyOpenCl Buffer object to facilitate reading and writing
    data to these.
    """
    ctx = attr.ib()
    queue = attr.ib()
    size = attr.ib()
    dtype = attr.ib()
    pgr_write = attr.ib(default=0)
    pgr_read = attr.ib(default=0)
    cl_buffer = attr.ib(init=False)
    host_array = attr.ib(init=False)
    nbytes = attr.ib(init=False)

    def __attrs_post_init__(self):
        assert self.pgr_write ^ self.pgr_read, \
            "pgr_write and pgr_read must be different"
        self.nbytes = memory_size(self.size, self.dtype)
        self.host_array = np.empty(self.size, dtype=np.dtype(self.dtype))
        if self.pgr_write:
            self.cl_buffer = self.get_read_buffer()
        elif self.pgr_read:
            self.cl_buffer = self.get_write_buffer()
        else:
            raise Exception("Did not create a buffer")

    def get_read_buffer(self):
        """
        Create and return a buffer object for the program to write to
        """
        return cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY, self.nbytes)

    def get_write_buffer(self):
        """
        Create and return a buffer object for the program to read from
        """
        return cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY, self.nbytes)

    def read(self):
        """
        Copy over the contents of the buffer and return them as a numpy array
        """
        cl.enqueue_copy(self.queue, self.host_array, self.cl_buffer)
        return self.host_array

    def write(self, x):
        """
        Write the given array over to the buffer
        """
        cl.enqueue_copy(self.queue, self.cl_buffer, x)

    def as_cl(self):
        """
        Return the OpenCL buffer object.
        """
        return self.cl_buffer


@attr.s
class MappedBuffer(Buffer):
    """
    Similar to Buffer but uses mapped buffers for when using OpenCL with
    a CPU.
    """
    def get_read_buffer(self):
        return self.get_write_buffer()

    def get_write_buffer(self):
        buffer = cl.SVM(cl.csvm_empty_like(self.ctx, self.host_array))
        with buffer.map_rw(self.queue) as host:
            self.host_array = host
        return buffer

    def read(self):
        return self.host_array[:]

    def write(self, x):
        self.host_array[:] = x


def memory_size(size, dtype_str):
    """
    Return the size of the array in bytes
    """
    return np.dtype(dtype_str).itemsize * size


def _calculate_system_size(target_stepsize):
    """
    Gets the number of discretisation points per side.
    """
    return int(np.ceil(2/target_stepsize))


def _calculate_spacial_step(system_size):
    """
    Gets the spacial step for a system of a given size per side
    """
    return 2/system_size


@attr.s
class InvertedOperator:
    operator = attr.ib()
    make_preconditioner = attr.ib(default=True)
    preconditioner = attr.ib(default=None)
    maxiter = attr.ib(default=100)
    # define container to contain information on solver result
    info = NamedTuple("info", error=list, status=int)

    def __attrs_post_init__(self):
        if self.make_preconditioner:
            self.preconditioner = self.operator.get_preconditioner()

    def diff(self, u, b):
        """
        Append the difference between the vectors operator@u and b to the history
        list.
        """
        return np.linalg.norm(
                self.operator @ u - b
            )

    def __matmul__(self, x):
        # define error logging callback for plotting convergence
        history = []
        def callback(u): history.append(self.diff(u, x))
        # run CG method
        u, status = cg(
            self.operator, x,
            callback=callback,
            M=self.preconditioner,
            maxiter=self.maxiter
        )
        # return tuple of solution and information on convergence
        return u, self.info(history, status)



if __name__ == "__main__":
################################################################################
########################## MAIN PROGRAM RUNS HERE ##############################
################################################################################

    # USE_MAPPED_BUFFER controls what type of buffer to use when
    # transfering data to hardware. Unmapped buffers do not share
    # host memory and instead copy over their data before executing
    # the kernel.
    # Do not use a mapped buffer if running the kernel on a GPU.
    USE_MAPPED_BUFFER = False

    # Initialise the specialised solver for this problem
    time_step = 1e-6
    spatial_step = 1e-3
    solver = Solver8(spatial_step, time_step)

    # Kick of the simulation stopping once the target temperature has been
    # bounded
    target_temperature = 1
    with Timer() as t:
        low, high, u = solver.stop_when_center_is(target_temperature)

    # Print results to console
    print('lower bound: %.4f @ %.4f s' % low)
    print('upper bound: %.4f @ %.4f s' % high)
    print('Total runtime: %.4f seconds' % t.interval)

    # Open figure for visual inspection
    solver.show(u, 'Final State')
