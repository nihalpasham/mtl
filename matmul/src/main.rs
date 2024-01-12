use std::{
    fmt::Debug,
    ops::{AddAssign, Index, IndexMut, Mul},
};

#[derive(Clone, Debug)]
pub struct Matrix<T> {
    rows: usize,
    cols: usize,
    elems: Vec<T>,
}

impl<T: Copy> Matrix<T> {
    pub fn new(rows: usize, cols: usize, elems: &[T]) -> Self {
        assert_eq!(
            elems.len(),
            rows * cols,
            "no: of elems must equal (rows * cols)"
        );
        Matrix {
            rows,
            cols,
            elems: elems.to_vec(),
        }
    }
    pub fn is_square(&self) -> bool {
        self.rows == self.cols
    }
}

impl<T> Index<(usize, usize)> for Matrix<T> {
    type Output = T;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        let (row, col) = index;
        let idx = self.cols * row + col;
        &self.elems[idx]
    }
}

impl<T> IndexMut<(usize, usize)> for Matrix<T> {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        let (row, col) = index;
        let idx = self.cols * row + col;
        &mut self.elems[idx]
    }
}

impl<T> Mul for Matrix<T>
where
    T: AddAssign + Mul<Output = T> + Copy + Default + Debug,
{
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        assert_eq!(self.cols, rhs.rows, "cant multiply matrices, check dims");
        // instantiate the resultant matrix
        let rows = self.rows;
        let cols = rhs.cols;
        let size = rows * cols;
        let elems: Vec<T> = (0..size).into_iter().map(|_| T::default()).collect();

        let mut result = Matrix::new(rows, cols, elems.as_slice());

        // for each row
        for row in 0..self.rows {
            // and for every column in the matrix
            for col in 0..rhs.cols {
                let mut tmp: T = T::default();
                // compute the sum of the dotprod of elems
                for elem in 0..self.rows {
                    tmp += self.elems[row * self.cols + elem] * rhs.elems[elem * rhs.cols + col];
                }
                result[(row, col)] = tmp;
            }
        }
        result
    }
}

fn main() {
    let mtx_a = Matrix::new(
        4,
        4,
        &[
            1., 2., 3., 4., 
            5., 6., 7., 8., 
            1., 2., 3., 4., 
            5., 6., 7., 8.,
        ],
    );
    let mtx_b = Matrix::new(
        4,
        3,
        &[
            5., 6., 7., //8., 
            1., 2., 3., //4., 
            5., 6., 7., //8., 
            1., 2., 3., //4.,
        ],
    );

    let res = mtx_a * mtx_b;
    println!("result: {:?}", res)
}
