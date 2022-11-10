import math

def surface_cercle(Rayon):
    if(Rayon > 0):
        surface = math.pi * pow(Rayon,2)
        return surface
    else:
        return 0

def surface_cercle_creux(Rayon_ext, Rayon_int):

    if(Rayon_int < Rayon_ext):
        surface_int = math.pi * pow(Rayon_int, 2)
        surface_ext = math.pi * pow(Rayon_ext, 2)
        surface = surface_ext - surface_int
        return surface
    else:
        return 0

def surface_rectangle(Base, Hauteur):
    if(Base > 0 and Hauteur > 0):
        surface = Base * Hauteur
        return surface
    else:
        return 0

def surface_rectangle_creux(Base_int, Hauteur_int, Base_ext, Hauteur_ext):
    if(Base_int < Base_ext and Hauteur_int < Hauteur_ext):
        surface_int = Base_int * Hauteur_int
        surface_ext =Base_ext * Hauteur_ext
        surface = surface_ext - surface_int
        return surface
    else:
        return 0

def surface_ipn(b, t, s, h):
    if(b > 0 and t > 0 and s>0 and 2*t<h):
        surface_h = b * t
        surface_v = s * (h - 2*t)
        surface = surface_h * 2 + surface_v
        return surface
    else:
        return 0