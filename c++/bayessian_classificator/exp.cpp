#include <iostream>

template <uint first, uint second>
void f(){
    std::cout << first << std::endl;
    std::cout << second << std::endl;
};

class A{
public:
    template <uint a, uint b>
    void f();
};

template <uint a, uint b>
void A::f()
{
    std::cout << a << std::endl;
    std::cout << b << std::endl;
}


int main(){
    A a;
    a.f<0,1>();
    return 0;
}