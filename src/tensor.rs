use std::{slice, sync::Arc, vec};
use std::ops::{Mul, Add};
use num::Zero;
 
pub struct Tensor<T> {
    data: Arc<Box<[T]>>,
    shape: Vec<usize>,
    offset: usize,
    length: usize,
}
 
// 多维张量转置子函数
fn compute_strides(shape: &Vec<usize>) -> Vec<usize> {
    let mut strides = vec![0; shape.len()];
    let mut stride = 1;
    for i in (0..shape.len()).rev() {
        strides[i] = stride;
        stride *= shape[i];
    }
    strides
}
 
fn compute_index(index: usize, strides: &Vec<usize>) -> Vec<usize> {
    let mut indexs = vec![0; strides.len()];
    let mut remainder = index;
    for i in 0..strides.len() {
        indexs[i] = remainder / strides[i];
        remainder %= strides[i];
    }
    indexs
}
 
fn compute_flat_index(indexs: Vec<usize>, strides: &Vec<usize>) -> usize{
    let mut flat_index: usize = 0;
    for i in 0..indexs.len(){
        flat_index += indexs[i] * strides[i];
    }
    flat_index
}
 
 
impl<T: Copy + Clone + Default> Tensor<T> {
    pub fn new(data: Vec<T>, shape: &Vec<usize>) -> Self {
        let length = data.len();
        Tensor {
            data: Arc::new(data.into_boxed_slice().try_into().unwrap()),
            shape: shape.clone(),
            offset: 0,
            length: length,
        }
    }
 
    pub fn default(shape: &Vec<usize>) -> Self {
        let length = shape.iter().product();
        let data = vec![T::default(); length];
        Self::new(data, shape)
    }
 
    pub fn data(&self) -> &[T] {
        &self.data[self.offset..][..self.length]
    }
 
    pub unsafe fn data_mut(&mut self) -> &mut [T] {
        let ptr = self.data.as_ptr().add(self.offset) as *mut T;
        slice::from_raw_parts_mut(ptr, self.length)
    }
 
    pub fn shape(&self) -> &Vec<usize> {
        &self.shape
    }
 
    pub fn size(&self) -> usize {
        self.length
    }
 
    pub fn clone(&self) -> Self {
        Tensor {
            data: self.data.clone(),
            shape: self.shape.clone(),
            offset: self.offset,
            length: self.length,
        }
    }
 
    // Reinterpret the tensor as a new shape while preserving total size.
    pub fn reshape(&mut self, new_shape: &Vec<usize>) -> &mut Self {
        let new_length: usize = new_shape.iter().product();
        if new_length != self.length {
            let old_shape = self.shape.clone();
            panic!("New shape {new_shape:?} does not match tensor of {old_shape:?}");
        }
        self.shape = new_shape.clone();
        self
    }
 
    pub fn slice(&self, start: usize, shape: &Vec<usize>) -> Self {
        let new_length: usize = shape.iter().product();
        assert!(self.offset + start + new_length <= self.length);
        Tensor {
            data: self.data.clone(),
            shape: shape.clone(),
            offset: self.offset + start,
            length: new_length,
        }
    }
 
    // 多维张量转置
    pub fn transpose(&self, perm: Vec<usize>) -> Self{
        let shape_len = self.shape.len();
        let data = self.data();
 
        let mut new_shape = vec![0; shape_len];
 
        for i in 0..shape_len {
            new_shape[i] = self.shape[perm[i]];
        }
 
        let old_strides = compute_strides(&self.shape());
 
        let new_strides = compute_strides(&new_shape);
 
        // 创建T数组
        let mut new_data = vec![T::default(); self.data.len()];
 
        for i in 0..self.size(){
            let old_index = compute_index(i, &old_strides);
 
            let mut new_index = vec![0; old_index.len()];
            
            for j in 0..old_index.len() {
                new_index[perm[j]] = old_index[j];
            }
 
            let new_flat_index = compute_flat_index(new_index, &new_strides);
 
            new_data[new_flat_index] = data[i];
        }
        Tensor::new(new_data, &new_shape)
    }
 
}
 
// Some helper functions for testing and debugging
impl Tensor<f32> {
    #[allow(unused)]
    pub fn close_to(&self, other: &Self, rel: f32) -> bool {
        if self.shape() != other.shape() {
            return false;
        }
        let a = self.data();
        let b = other.data();
        
        return a.iter().zip(b).all(|(x, y)| float_eq(x, y, rel));
    }
    #[allow(unused)]
    pub fn print(&self){
        println!("shpae: {:?}, offset: {}, length: {}", self.shape, self.offset, self.length);
        let dim = self.shape()[self.shape().len() - 1];
        let batch = self.length / dim;
        for i in 0..batch {
            let start = i * dim;
            println!("{:?}", &self.data()[start..][..dim]);
        }
    }
}
 
#[inline]
pub fn float_eq(x: &f32, y: &f32, rel: f32) -> bool {
    (x - y).abs() <= rel * (x.abs() + y.abs()) / 2.0
}