# This script calculates and prints the answer to Problem 2 of the SIAM 100
# digit challenge. It also generates and displays in a plot the path of this
# particle.
using Base: BigFloat, Number
using LinearAlgebra: norm, dot
using Plots: partialcircle, display, plot, plot!, scatter!, unzip, ylims!, xlims!

# useful information structures:
struct Mirror
    pos::Array
    rad::Number
end

struct Photon
    pos::Array
    vel::Array
end


"""
    coefs(photon)

Return a tuple of the equivalent slope and y-intercept of the line drawn by the
given travelling Photon.
"""
function coefs(photon::Photon)
    slope = photon.vel[2]/photon.vel[1]
    intersect = photon.pos[2] - photon.pos[1] * photon.vel[2] / photon.vel[1]
    return slope, intersect
end


"""
    intersection(photon, mirror)

Compute and return the intersection point of a travelling photon and a mirror.
If there is no intersection, it returns an array of NaN.

# Returns
- `Array`: representing the coordinates of the intersection
- `boolean`: indicating if there is an intersection or not
"""
function intersection(photon::Photon, mirror::Mirror)
    # coefficients related to path of photon
    m,b = coefs(photon)
    c2 = m^2 + 1                                  # coefficient for x^2
    c1 = 2(m*(b - mirror.pos[2]) - mirror.pos[1]) # coefficient for x^1
    c0 = (b - mirror.pos[2])^2 + mirror.pos[1]^2 - mirror.rad^2
    f(x) = m*x + b

    Δ = c1^2 - 4*c2*c0
    if Δ < 0
        return [NaN, NaN], false
    elseif Δ == 0
        x = -c1/(2 *c2)
        y = f(y)
        return [x, y], true
    else
        x1 = (-c1 + sqrt(Δ))/(2c2)
        y1 = f(x1)
        p1 = [x1, y1]
        d1 = distance(photon.pos, p1)
        x2 = (-c1 - sqrt(Δ))/(2c2)
        y2 = f(x2)
        p2 = [x2, y2]
        d2 = distance(photon.pos, p2)
        # println("distance ", d1, " ", d2, " ", d1 < d2)
        return d1 < d2 ? (p1, true) : (p2, true)
    end
end


"""
    distance(pos0, pos1)

Return the distance between the two Positions
"""
function distance(a, b)
    return sqrt((a[1] - b[1])^2 + (a[2] - b[2])^2)
end


"""
    plot_journey!(position_1, position_2)

Plot onto the existing graph a line between the two given points.
"""
function plot_journey!(position_1, position_2)
    plot!(
        [position_1[1], position_2[1]],
        [position_1[2], position_2[2]]
    )

end


"""
    plot_mirror!(i, j)

Plot onto the existing plot, at the given coordinates a cirlce representing one
of the mirrors.
"""
function plot_mirror!(i, j)
    mirror = Mirror([i, j], 1//3)
    pts = get_circle_pts(mirror)
    plot!(pts)
end


"""
    get_circle_pts(m::Mirror)

Returns a set of points to be used for plotting a the circles representing the
mirrors
"""
function get_circle_pts(m::Mirror)
    pts = partialcircle(0, 2π, 100, m.rad)
    x, y = unzip(pts)
    x .+= m.pos[1]
    y .+= m.pos[2]

    return collect(zip(x, y))
end


"""
    get_closest(reference, option_1, option_2)

Of the two given points as options, return the one that closest to the reference
point.
"""
function get_closest(reference, option_1, option_2)
    distance_1 = distance(reference, option_1)
    distance_2 = distance(reference, option_2)

    return distance_1 < distance_2 ? option_1 : option_2
end


"""
    reflection(particle::Photon, mirror::Mirror)

Returns a tuple, the first element corresponding to the new photon after it has
been relected off the mirror, the second element is flag that indicates if there
is an intersection between the given particle and mirror
"""
function reflection(particle::Photon, mirror::Mirror)
    point, intersect_flag = intersection(particle, mirror)
    radial_line =  point - mirror.pos
    n = radial_line/norm(radial_line)
    velocity = particle.vel - 2*dot(particle.vel, n) * n

    return Photon(point, velocity), intersect_flag
end


"""
    get_range_values(particle::Photon, m)

Returns a start and end value between which to search for intersections as well
as the direction in which to search (ascending or descending)
"""
function get_range_values(particle::Photon, m)
    if abs(m) > 1
        index = 2
    else
        index = 1
    end

    if !isnan(particle.vel[index])
        direction = sign(particle.vel[index])
    else
        direction = 1
    end
    if direction > 0
        start = ceil(particle.pos[index])
    else
        start = floor(particle.pos[index])
    end
    if isnan(start)
        throw("NaN for start")
    end
    # iterate through y's and get x's
    final = 20*direction

    return start, direction, final
end


"""
    next_photon(particle::Photon)

From a given particle (position and speed), returns the next photon in the
series of reflections
"""
function next_photon(particle::Photon)
    m, b = coefs(particle)

    init, direction, final = get_range_values(particle, m)

    for z=init:direction:final
        if abs(m) > 1
            y = z
            x = (y - b)/m

            position_a = [ceil(x), y]
            position_b = [floor(x), y]

        else
            x = z
            y = m*x + b
            position_a = [x, ceil(y)]
            position_b = [x, floor(y)]
        end

        position = get_closest([x, y], position_a, position_b)

        new_photon, reflected = reflection(
            particle,
            Mirror(position, 1//3)
        )
        if reflected
            return new_photon
        end

    end
    throw("No detections after many mirrors")
end


"""
    position_after(particle, time)

Returns the position of the particle after an amount of time has passed,
assuming that it keeps traveling in a straight trajectory.
"""
function position_after(particle, time)
    return particle.pos + particle.vel * time
end

################################################################################
############################## Main Simulation #################################
################################################################################


## Plotting Setup
lim = 2 # limits of axes in plot
plot(aspect_ratio=:equal, legend=false)
ylims!((-lim, lim))
xlims!((-lim, lim))

# plot mirrors
for i=-lim:1:lim, j=-lim:1:lim
    plot_mirror!(i, j)
end


let
    # Declare the initial setup
    particle = Photon([BigFloat("0.5"), BigFloat("0.1")], [1, 0])
    d = BigFloat(0)

    ## Iterate through relections
    for i=1:15
        next = next_photon(particle)
        Δd = distance(particle.pos, next.pos)
        # if the next particle would reflect after the target time has passed
        # break out of the loop early
        if d + Δd >= 10
            break
        end
        # plot this leg of the journey
        plot_journey!(particle.pos, next.pos)
        # increment distance
        d += Δd
        particle = next
    end

    # find position at target time along final leg
    Δt = 10 - d # remaining time
    final_pos = position_after(particle, Δt)
    plot_journey!(particle.pos, final_pos)
    println(distance([0, 0], final_pos))
end

plot!()
