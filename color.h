//color.h
#ifndef _COLOR_H_
#define _COLOR_H_
class Color
{
public:
    Color(double val) {r = g = b = val;}
    Color(double red, double green, double blue);
    Color(const Color & c) {*this = c;}
    
    void R(double red)  {r = red;}
    void G(double green) {g = green;}
    void B(double blue)  {b = blue;}
    
    double R() const {return r;}
    double G() const {return g;}
    double B() const {return b;}
    
    void clamp();
    //define some operators for this class:
    Color& operator=(const Color& rhs);
    friend Color operator * (const Color& c, double f);
    friend Color operator * (double f, const Color& c);
    friend Color operator * (const Color& c1, const Color& c2);
    friend Color operator / (const Color& c, double f);
    friend Color operator + (const Color& c1, const Color& c2);
    //private:
    //Data memebrs are not private because of performance hits. 
    //Its better to access directly in critical cases than to use functions!    
    double r;
    double g;
    double b;

    float maxComponent();
};
#endif
