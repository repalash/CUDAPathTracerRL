//color.cpp
#include "color.h"

Color::Color(double red, double green, double blue)
{
    r = red;
    b = blue;
    g = green;
}

Color& Color::operator=(const Color& rhs)
{
r = rhs.r;
g = rhs.g;
b = rhs.b;
return *this;
}

void Color::clamp()
{
    if(r>1.0f) r = 1.0f;
    if(g>1.0f) g = 1.0f;
    if(b>1.0f) b = 1.0f;
    
    if(r<0.0f) r = 0.0f;
    if(g<0.0f) g = 0.0f;
    if(b<0.0f) b = 0.0f;
}

Color operator * (const Color& c, double f)
{ return Color(c.r*f, c.g*f, c.b*f);}

Color operator * (double f, const Color& c)
{ return Color(c.r*f, c.g*f, c.b*f);}

Color operator * (const Color& c1, const Color& c2)
{return Color(c1.r*c2.r, c1.g*c2.g, c1.b*c2.b);}

Color operator / (const Color& c, double f)
{return Color(c.r/f, c.g/f, c.b/f);}

Color operator + (const Color& c1, const Color& c2)
{return Color(c1.r+c2.r, c1.g+c2.g, c1.b+c2.b);}

float Color::maxComponent() {
    return (float) (r > g ? r > b ? r : b : g > b ? g : b);
}


