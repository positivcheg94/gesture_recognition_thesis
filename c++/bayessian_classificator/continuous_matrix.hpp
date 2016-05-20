#ifndef CONTINUOUS_MATRIX_HPP
#define CONTINUOUS_MATRIX_HPP

#include <iostream>
#include <cstring>

#include <boost/serialization/version.hpp>
#include <boost/serialization/split_member.hpp>


template<class T,class S = size_t>
class continuous_matrix {

    friend class boost::serialization::access;

    typedef T value_type;

    typedef T* pointer;
    typedef const pointer const_pointer;

    typedef value_type& reference;
    typedef const reference const_reference;

    typedef S size_type;

    typedef pointer iterator;
    typedef const pointer const_iterator;

    size_type _rows;
    size_type _cols;
    size_type _total;
    pointer _data;

    template<class Archive>
    void save(Archive & ar, const unsigned int version) const
    {
        ar & _rows;
        ar & _cols;
        ar & _total;
        auto elem = cbegin(), last = cend();
        for(;elem != last; elem++)
            ar & *elem;
    }

    template<class Archive>
    void load(Archive & ar, const unsigned int version)
    {
        ar & _rows;
        ar & _cols;
        ar & _total;

        if (_data != nullptr ){
            delete[] _data;
        }
        _data = new value_type[_total];

        auto elem = cbegin(), last = cend();
        for(;elem != last; elem++)
            ar & *elem;
    }

    BOOST_SERIALIZATION_SPLIT_MEMBER()
public:

    continuous_matrix() : _data(nullptr), _rows(0), _cols(0), _total(0) { }

    continuous_matrix(size_type rows, size_type cols) : _rows(rows), _cols(cols), _total(rows * cols) {
        _data = new value_type[_total];
    }

    continuous_matrix(size_type rows, size_type cols, const_pointer data) : _rows(rows), _cols(cols), _total(rows * cols) {
        _data = new value_type[_total];
        std::memcpy(_data, data, _total * sizeof(value_type));
    }

    continuous_matrix(const continuous_matrix &matrix) : _rows(matrix._rows), _cols(matrix._cols), _total(matrix._rows * matrix._cols) {
        _data = new value_type[_total];
        std::memcpy(_data, matrix._data, _total * sizeof(value_type));
    }

    continuous_matrix(continuous_matrix && matrix) : _rows(matrix._rows), _cols(matrix._cols), _total(matrix._rows * matrix._cols) {
        _data = matrix._data;
        matrix._rows = matrix._cols = matrix._total = 0;
        matrix._data = nullptr;
    }

    inline reference operator()(size_type row, size_type col) {
        return _data[row * _cols + col];
    }

    inline reference get(size_type row, size_type col) const {
        return _data[row * _cols + col];
    }

    inline void set(size_type row, size_type col, const_reference value) {
        _data[row * _cols + col] = value;
    }

    void increment(size_type row, size_type col){
        _data[row * _cols + col]++;
    }

    ~continuous_matrix() {
        if (_data != nullptr)
            delete[] _data;
    }

    inline size_type rows(){
        return _rows;
    }

    inline size_type cols(){
        return _cols;
    }

    inline pointer data() {
        return _data;
    }

    inline iterator begin() {
        return _data;
    }

    inline iterator end() {
        return _data+_total;
    }

    inline const_iterator cbegin() const {
        return _data;
    }

    inline const_iterator cend() const {
        return _data+_total;
    }

    void resize(size_type rows, size_type cols){
        _rows = rows;
        _cols = _cols;
        _total = _rows*_cols;
        if ( _data != nullptr )
            delete[] _data;
        _data = new value_type[_total];
    }


};

typedef continuous_matrix<double> dmatrix;
typedef continuous_matrix<size_t> umatrix;

#endif //CONTINUOUS_MATRIX_HPP
