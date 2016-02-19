// Data structures
struct Camera {
    vec3 pos;
    float fov_factor;
    vec2 res;
};
struct Ray {
    vec3 dir;
    vec3 origin;
};
struct Sphere {
    vec3 pos;
    float rad;
    int mt;
};

struct Box {
    vec3 min, max;
    int mt;
};

// Water plane
// H(P) = Ax + By + Cz + D = n P + D = 0
struct WaterPlane {
    vec3 norm;
    float D;
    int mt;
};

struct Hit {
    vec3 pos;
    vec3 norm;
    int mt;  // material
	float R;
};
struct Light {
    vec3 pos; //positional light use pos while directional light use direction
    float size; //radius for sphere light
    float Is; //specular intensity
    float Id; //diffuse intensity
};

// Textures
uniform sampler2D prev_tex;

// Wall textures
uniform sampler2D front_wall_tex;
uniform sampler2D back_wall_tex;
uniform sampler2D left_wall_tex;
uniform sampler2D right_wall_tex;
uniform sampler2D ceil_wall_tex;

// Pool texture
uniform sampler2D pool_tex;

// Material texture table (Ka, Kd, Ks)
uniform sampler2D mtl_tex;

// Water normal
uniform sampler2D water_norm_0;
uniform sampler2D water_norm_1;



// Camera and transformation matrix
uniform Camera camera;
uniform mat3 trans;

// Time and control
uniform float glob_time;
uniform float sample_count;
uniform int pause;
uniform int refresh;

uniform float mtl_num;

varying vec2 uv;

// Constants
const int MAX_BOUNCE_NUM = 6; //max bounce time
const float EPSILON = 0.0001;
const float INFINITY = 1000000.;
const float x_pos = 1.7;
const float y_pos = 0.0;
const float PI = 3.1415;
const float BASE_ATTR = 100.0;
const int X = 0, Y = 1, Z = 2;
const int LIGHT_NUM = 4;

const float SAMPLE_THRESHOLD = 100.0;
const float MTL_STRIDE = 4.0;

// Texture constants
const int DEFAULT_TEX = 7;
const int FRONT_WALL_TEX = 1;
const int BACK_WALL_TEX = 9;
const int LEFT_WALL_TEX = 3;
const int RIGHT_WALL_TEX = 8;
const int POOL_TEX = 5;
const int CEIL_TEX = 2;
const int WATER_TEX = 0;
const vec3 water_color = vec3(1,1,1);

const int REFLECTIVE = 3;

// Global variables
float KA = 0., KD = 1. / MTL_STRIDE, KS = 2. / MTL_STRIDE, ATTR = 1.; // Texture lookup indices
vec3 rand_vec = vec3(0);
int  cur_bounce_num  = 0;
bool hit_water = false;

// Water Plane Parameter
float area_length, d_h; // plane area length
float eta; // the ratio of refraction
WaterPlane water;

// Light
Light lights[LIGHT_NUM];

// The room
const Box room = Box(vec3(-10, -10, -20), vec3(10, 10, 20), 1);

// The spheres
const int SPHERE_NUM = 3;
Sphere sphere[SPHERE_NUM];

// Cubes
const int CUBE_NUM = 1;
Box cubes[CUBE_NUM];

// functions declaration
vec3 calc_light_color(in Hit hit, in vec3 N, inout Ray eye_ray); 
vec2 get_room_uv(vec3 pos, int axis);
vec3 get_box_normal(vec3 hit, in Box box);
bool hit_objects(in Ray eye_ray, out Hit hit, bool once);
vec2 intersect_cube(in Ray eye_ray, Box box, out float dist);
bool intersect_water(in Ray eye_ray, in WaterPlane plane, out float dist);


// utility functions
float hash( float n ) {
    return fract(sin(n) * 43758.5453);
}

// Noise function taken from http://stackoverflow.com/questions/4200224/random-noise-functions-for-glsl
// The noise function returns a value in the range -1.0f -> 1.0f
float noise( in vec3 x ) {
    vec3 p = floor(x);
    vec3 f = fract(x);

    f = f * f * (3.0 - 2.0 * f);
    float n = p.x + p.y * 57.0 + 113.0 * p.z;
    return mix(mix(mix( hash(n +  0.0), hash(n +  1.0), f.x),
                   mix( hash(n + 57.0), hash(n + 58.0), f.x), f.y),
               mix(mix( hash(n + 113.0), hash(n + 114.0), f.x),
                   mix( hash(n + 170.0), hash(n + 171.0), f.x), f.y), f.z);
}
// Random function from http://madebyevan.com/webgl-path-tracing/webgl-path-tracing.js
// result: [0, 1]
float random(vec3 scale, float seed) {
    return fract(sin(dot(gl_FragCoord.xyz + seed, scale)) * 43758.5453 + seed);
}

// Random function from http://madebyevan.com/webgl-path-tracing/webgl-path-tracing.js
// Get a normalized random direction
vec3 get_uniform_rand_dir(float seed) {
    float u = random(vec3(12.9898, 78.233, 151.7182), seed);
    float v = random(vec3(63.7264, 10.873, 623.6736), seed);
    float z = 1.0 - 2.0 * u;
    float r = sqrt(1.0 - z * z);
    float angle = 6.283185307179586 * v;
    return vec3(r * cos(angle), r * sin(angle), z);
}
// Function from http://madebyevan.com/webgl-path-tracing/webgl-path-tracing.js
vec3 get_rand_vec(float seed) {
    return get_uniform_rand_dir(seed) * sqrt(random(vec3(36.7539, 50.3658, 306.2759), seed));
}

// Function from http://www.rorydriscoll.com/2009/01/07/better-sampling/
// Get the diffuse direction using cosine weighted method, not random
vec3 get_diffuse_dir(in float seed, in vec3 normal) {
    float u = random(vec3(12.9898, 78.233, 151.7182), seed);
    float v = random(vec3(63.7264, 10.873, 623.6736), seed);
    float r = sqrt(u);
    float angle = 6.283185307179586 * v;
    // compute basis from normal
    vec3 sdir, tdir;
    if (abs(normal.x) < .5) {
        sdir = cross(normal, vec3(1, 0, 0));
    } else {
        sdir = cross(normal, vec3(0, 1, 0));
    }
    tdir = cross(normal, sdir);
    return r * cos(angle) * sdir + r * sin(angle) * tdir + sqrt(1. - u) * normal;
}

float ambient(float coe) {
    return coe;
}
float diffuse(vec3 L, vec3 N, float Id) {
    return max(dot(L, N), 0.) * Id;
}
float specular(vec3 R, vec3 V, float Is, float Ns) {
    return pow(max(dot(R, V), 0.), Ns) * Is;
}

// Get the texture coord of room
vec2 get_room_uv(vec3 pos, int axis) {
    vec2 result = pos.xy;
    if (axis == X) {
        pos.y /= room.max.y - room.min.y;
        pos.z /= room.max.z - room.min.z;
        pos += 0.5;
        result = vec2(pos.zy);
    } else if (axis == Y) {
        pos.x /= room.max.x - room.min.x;
        pos.z /= room.max.z - room.min.z;
        pos += 0.5;
        result = vec2(pos.xz);
    } else if (axis == Z) {
        pos.x /= room.max.x - room.min.x;
        pos.y /= room.max.y - room.min.y;
        pos += 0.5;
        result = vec2(pos.xy);
    }
    return result;
}

// Calculate the intersection of a ray and the House (box)
vec2 intersect_cube(in Ray eye_ray, Box box, out float dist) { //ray box intersect
	vec3 tMin = (box.min - eye_ray.origin) / eye_ray.dir;
    vec3 tMax = (box.max - eye_ray.origin) / eye_ray.dir;
    vec3 t1 = min(tMin, tMax);
    vec3 t2 = max(tMin, tMax);
    dist = INFINITY;
    float tNear = max(max(t1.x, t1.y), t1.z);
    float tFar = min(min(t2.x, t2.y), t2.z);
    dist = tFar;
    return vec2(tNear, tFar);
}

vec3 get_box_normal(vec3 hit, in Box box) {
    if(hit.x <= box.min.x + EPSILON )
        return vec3(-1.0, 0.0, 0.0);//box left
    else if(hit.x >= box.max.x -  EPSILON)
        return vec3(1.0, 0.0, 0.0);//box right
    else if(hit.y <= box.min.y +  EPSILON )
        return vec3(0.0, -1.0, 0.0);//box bottom
    else if(hit.y >= box.max.y - EPSILON)
        return vec3(0.0, 1.0, 0.0);//box top
    else if(hit.z <= box.min.z + EPSILON )
        return vec3(0.0, 0.0, -1.0);//box front
    else
        return vec3(0.0, 0.0, 1.0);//box back
}
int get_room_mtl(Hit hit) { //set material for bounding box
    int tex = DEFAULT_TEX;
    if(int(hit.norm.x) == -1) {
        tex = RIGHT_WALL_TEX;
    } else if (int(hit.norm.x) == 1) {
        tex = LEFT_WALL_TEX;
    } else if(int(hit.norm.y) == 1) {
        tex = POOL_TEX;
    } else if (int(hit.norm.y) == -1) {
        tex = CEIL_TEX;
    } else if (int(hit.norm.z) == -1) {
        tex = FRONT_WALL_TEX;
    } else if (int(hit.norm.z) == 1) {
        tex = BACK_WALL_TEX;
    }
    return tex;
}


// Ray water intersect
// d: dist, Ro: Ray origin, Rd: Ray direction, n: plane.norm
// P(d) = Ro + d * Rd, H(P) = n·P + D = 0, water plane
// n·(Ro + d * Rd) + D = 0
// d = -(D + n·Ro) / n·Rd
bool intersect_water(in Ray eye_ray, in WaterPlane plane, out float dist) {
    vec3 n = normalize(plane.norm);
    float D = plane.D;

    dist = INFINITY;
    // If it is under water, reverse the normalize and D
   // if (dot(n, -eye_ray.dir) < EPSILON) {
   //     D -= d_h / n.y;
   //     n = -n;
   //     D = -D;
    //} else {
        D += d_h / n.y;
    //}
    dist = -(D + dot(n, eye_ray.origin)) / dot(n, eye_ray.dir);
    if (dist < 0.)
        return false;
    // Displacement
    float deltal = 0;//noise(0.05 * eye_ray.dir * dist + 0.01 * glob_time);
    dist += deltal;
    d_h = eye_ray.dir.y * deltal;
    return true;
}

// Map the hit position to texture coord
vec2 get_water_tex_coord(vec3 pos) {
    return 0.5 * vec2((pos.x + area_length) / area_length, (pos.z + area_length) / area_length);;
}

// Get the water norm from the two normal texture
vec3 get_water_norm(vec3 pos) {
    vec2 uv = get_water_tex_coord(pos);
    float speed = 0.005;
    vec2 uv0 = vec2(fract(uv.x - speed * glob_time), fract(uv.y + speed * glob_time));
    vec2 uv1 = vec2(fract(uv.x + speed * glob_time), fract(uv.y - speed * glob_time));
    // vec2 uv1 = uv;
    return normalize(texture2D(water_norm_0, uv0));// + texture2D(water_norm_0, uv1) - 1.).xzy;
}

bool intersect_sphere(in Sphere sphere, in Ray eye_ray, out float dist) {
    vec3 c = sphere.pos - eye_ray.origin;
    float b = dot(eye_ray.dir, c);
    dist = INFINITY;
    if(b < 0.0)
        return false;
    float d = dot(c, c) - b * b;
    if(d < 0.0 || d > sphere.rad * sphere.rad)
        return false;
    dist = b - sqrt(sphere.rad * sphere.rad - d);
    return true;
}

bool hit_objects(in Ray eye_ray, out Hit hit, bool once) {
    float min_dist = INFINITY, dist = INFINITY;
    //hit cubes
    for (int i = 0; i < CUBE_NUM; i++) {
        vec2 cube_dist = intersect_cube(eye_ray, cubes[i], dist);
        if (cube_dist.x > 0.0
                && cube_dist.x < cube_dist.y
                && cube_dist.x < min_dist) {
            if (once)
                return true;
            min_dist = cube_dist.x;
            hit.mt = cubes[i].mt;
            hit.pos = eye_ray.origin + min_dist * eye_ray.dir;
            hit.norm = -get_box_normal(hit.pos, cubes[i]);
        }
    }
    // Hit spheres
    for(int i = 0; i < SPHERE_NUM; i++) {
        intersect_sphere(sphere[i], eye_ray, dist);
        if(dist < min_dist) {
            if(once) //shadow ray
                return true;
            min_dist = dist;
            hit.mt = sphere[i].mt;
            hit.pos = eye_ray.origin + min_dist * eye_ray.dir;
			hit.R=sphere[i].rad;
            hit.norm = normalize(hit.pos - sphere[i].pos);
        }
    }
    // Water do not block light
    if(once)
        return false;

    if(intersect_water(eye_ray, water, dist) && dist < min_dist && dist > 1.) {
        min_dist = dist;
        hit.pos = eye_ray.origin + dist * eye_ray.dir;
        //room as bounding box
        if(hit.pos.x < room.min.x || hit.pos.x > room.max.x || hit.pos.z < room.min.z || hit.pos.z > room.max.z)
            return false;
        if(cur_bounce_num == 0)
            hit_water = true;
        hit.norm = get_water_norm(hit.pos);
        //if (eye_ray.origin.y <  - water.D / water.norm.y + d_h)
        //    hit.norm = - hit.norm;8
        hit.mt = 20;//water.mt;
    }
    if(min_dist == INFINITY)
        return false;
    return true;
}

// Calculate light at a object point
vec3 calc_light_color(in Hit hit, in vec3 N, inout Ray eye_ray) {
    vec3 total_color = vec3(0); //accumulated color
    vec3 ka = vec3(0), kd = vec3(0), ks = vec3(0);
    vec3 attr = vec3(0, 1, 0);
    float alpha = 1., Ns = 0.; 
    int illum = 0; //material type
    float mtl_ind = float(hit.mt) / mtl_num; //change material index to uv coord

    //get matieral information from texture
    //get ka, kd, ks, ns according to different material
    ka = texture2D(mtl_tex, vec2(KA, mtl_ind)).xyz;
    kd = texture2D(mtl_tex, vec2(KD, mtl_ind)).xyz;
    attr = texture2D(mtl_tex, vec2(ATTR, mtl_ind)).xyz;	// x->illum y->Ns z->alpha

    // material type: 1 diffuse, 2 reflective, 3 transparent
    illum = int(attr.x * BASE_ATTR);
    Ns = (attr.y * BASE_ATTR);
    alpha = attr.z;

    // Get room texture
    vec2 tex_coord;
    if (hit.mt == FRONT_WALL_TEX) {
        // front wall
        tex_coord = get_room_uv(hit.pos, Z);
		
        ka = kd = texture2D(front_wall_tex, tex_coord).rgb*0.85;//*1.6;
    } else if (hit.mt == BACK_WALL_TEX) {
        // back wall
        tex_coord = get_room_uv(hit.pos, Z);
        ka = kd = texture2D(back_wall_tex, tex_coord).rgb*0.85;//*0.695;
    } else if(hit.mt == LEFT_WALL_TEX) {
        // left wall
        tex_coord = get_room_uv(hit.pos, X);
        ka = kd = texture2D(left_wall_tex, tex_coord).rgb;//*1.114;
    } else if (hit.mt == RIGHT_WALL_TEX) {
        // right wall
        tex_coord = get_room_uv(hit.pos, X);
        ka = kd = texture2D(right_wall_tex, tex_coord).rgb;//*1.114;
    } else if(hit.mt == POOL_TEX) {
        // pool
        tex_coord = get_room_uv(hit.pos, Y);
        ka = kd = texture2D(pool_tex, tex_coord).rgb*0.91;//*0.87;
    } else //(hit.mt == CEIL_TEX) 
	{
		tex_coord = get_room_uv(hit.pos, Y);
        ka = kd = texture2D(ceil_wall_tex, tex_coord).rgb*0.91;//*1.4;
    }

    if(illum == REFLECTIVE) {
        // reflective material such as metal
        ks = texture2D(mtl_tex, vec2(KS, mtl_ind)).xyz; //specular light coefficient
        // if(alpha < 1.0 - EPSILON && mod(int(glob_time), 5) == 0) {
        if(alpha < 1.0 - EPSILON) { 
            //transparent
            //total_color += water_color * 0.8;
         //hit.norm +=  0.1*rand_vec;
		 //hit.norm=normalize(-hit.norm);
		 hit.pos=hit.pos-hit.R*2*N;
            eye_ray.dir = refract(eye_ray.dir,N,  0.3);
			//.dir=normalize(normalize(eye_ray.dir)+normalize(rand_vec));
        } else {
            // reflect
            eye_ray.dir = reflect(eye_ray.dir, N);
			//total_color += water_color * 0.01;
			//total_color = 1.1*total_color;
			//eye_ray.dir = refract(eye_ray.dir, N, eta);
        }
    } else if(hit.mt ==20){
        //eye_ray.dir = get_diffuse_dir(sample_count + float(cur_bounce_num), N);
			total_color += 0.005*water_color;
			//total_color += water_color *0.04*(hit.pos.y)/20;
         eye_ray.dir = N;
         //eye_ray.dir = reflect(eye_ray.dir, N);
		
	}
		else{
        eye_ray.dir = get_diffuse_dir(sample_count + float(cur_bounce_num), N);
         eye_ray.dir = N + get_rand_vec(sample_count + float(cur_bounce_num));
         eye_ray.dir = reflect(eye_ray.dir, -N);
    }
    eye_ray.dir = normalize(eye_ray.dir);

    //diffuse material without specular light
   // if (camera.pos.y < - water.D / water.norm.y + d_h) {
    //    ka = .2 * ka + .8 * texture2D(mtl_tex, vec2(KA, float(WATER_TEX) / mtl_num)).xyz;
    //}

    // Ambient light
    total_color += ambient(0.1) * ka * attr.z;

    vec3 L, R;
    vec3 V = - normalize(eye_ray.origin - hit.pos);
    float attenuation = 1.;
    Ray shadow_ray;
    Hit shadow_hit;
    eye_ray.origin = hit.pos;

    for(int i = 0; i < LIGHT_NUM; i++) {
        L = lights[i].pos - hit.pos;
        L = normalize(L);
        R = reflect(L, N);
        //shadow
		if(i==0){
			vec3 light_dir = lights[i].pos  - hit.pos +rand_vec;//* lights[i].size;
			shadow_ray = Ray(normalize(light_dir), hit.pos);
			if(hit_objects(shadow_ray, shadow_hit, true)) {
				return total_color;
			}
		}
        alpha *= attr.z * attenuation;
        total_color += diffuse(L, N, lights[i].Id ) * kd * alpha;
        total_color += specular(R, V, lights[i].Is, Ns) *  ks;
    }
    return total_color;
}


void init() {
    // Initialize spheres
    // sphere[0] = Sphere(vec3(-5, -8, -16), 2., 0);
    // sphere[1] = Sphere(vec3(-5, -8, -12), 2., 0);
    // sphere[2] = Sphere(vec3(-5, -8, -8), 2., 0);
    // sphere[3] = Sphere(vec3(-1.53589, -8, -10), 2., 0);
    // sphere[4] = Sphere(vec3(-1.53589, -8, -14), 2., 0);
    // sphere[5] = Sphere(vec3(1.9282, -8, -12), 2., 0);
    // sphere[6] = Sphere(vec3(-3.8453, -4.734, -10), 2., 2);
    // sphere[7] = Sphere(vec3(-3.8453, -4.734, -14), 2., 2);
    // sphere[8] = Sphere(vec3(-0.3812, -4.734, -12), 2., 2);
    // sphere[9] = Sphere(vec3(-2.6906, -1.468, -12), 2., 0);
    // sphere[0] = Sphere(vec3(4.0+x_pos, -1.5, 5.1547+y_pos), 1., 0);
    // sphere[1] = Sphere(vec3(3.0+x_pos, -1.5, 3.42265+y_pos), 1., 0);
    // sphere[2] = Sphere(vec3(5.0+x_pos, -1.5, 3.42265+y_pos), 1., 0);
    // sphere[3] = Sphere(vec3(4.0+x_pos, 0.133, 4.0+y_pos), 1., 2);
    sphere[0] = Sphere(vec3(-6.0 + x_pos, -3.0, 2.1547 + y_pos), 2, 6);
    sphere[1] = Sphere(vec3(2.0 + x_pos, -1.5, 3.42265 + y_pos), 1., 0);
    sphere[2] = Sphere(vec3(-0 + x_pos, -1.5, 3.42265 + y_pos), 1., 0);
    //sphere[3] = Sphere(vec3(4.0+x_pos, -1.33, 4.0+y_pos), 1., 7);

    // cubes[0] = Cube(vec3(0, -8, 0), vec3(4, -4, 10), 0);
    //cubes[0] = Box(vec3(0.0 + x_pos, -10, 0.0 + y_pos), vec3(0.6 + x_pos, -3, 0.6 + y_pos), 0);
    //cubes[0] = Box(vec3(7.4 + x_pos, -10, 0.0 + y_pos), vec3(8.0 + x_pos, -3, 0.6 + y_pos), 0);
    //cubes[2] = Box(vec3(0.0 + x_pos, -10, 7.4 + y_pos), vec3(0.6 + x_pos, -3, 8.0 + y_pos), 0 );
    //cubes[3] = Box(vec3(7.4 + x_pos, -10, 7.4 + y_pos), vec3(8.0 + x_pos, -3, 8.0 + y_pos), 0);
    cubes[0] = Box(vec3(-3.3 + x_pos, -9.99, -0.3 + y_pos), vec3(1.3 + x_pos, -2.5, 8.3 + y_pos), 0);
/*
    cubes[5] = Box(vec3(1.0 + x_pos, -10.0, 2.5 + y_pos), vec3(1.3 + x_pos, -6.0, 2.8 + y_pos), 2);
    cubes[6] = Box(vec3(1.0 + x_pos, -10.0, 5.5 + y_pos), vec3(1.3 + x_pos, -6.0, 5.8 + y_pos), 2);
    cubes[7] = Box(vec3(-2.0 + x_pos, -10, 2.3 + y_pos), vec3(-1.7 + x_pos, 0.0, 2.6 + y_pos), 2 );
    cubes[8] = Box(vec3(-2.0 + x_pos, -10, 5.7 + y_pos), vec3(-1.7 + x_pos, 0.0, 6.0 + y_pos), 2);
    cubes[9] = Box(vec3(-2.0 + x_pos, -5.7, 3.5 + y_pos), vec3(-1.7 + x_pos, 0.0, 3.8 + y_pos), 2);
    cubes[10] = Box(vec3(-2.0 + x_pos, -5.7, 4.5 + y_pos), vec3(-1.7 + x_pos, 0.0, 4.8 + y_pos), 2);
    cubes[11] = Box(vec3(-2.0 + x_pos, -6.0, 2.3 + y_pos), vec3(1.5 + x_pos, -5.7, 6.0 + y_pos), 2);
    cubes[12] = Box(vec3(-2.0 + x_pos, 0.0, 2.1 + y_pos), vec3(-1.7 + x_pos, 0.3, 6.2 + y_pos), 2);*/
    // Initialize lights
    lights[0] = Light(vec3(0, 22, 27), 10, .05, .05);
	lights[1] = Light(vec3(0, -22, -27), 10, .05, .05);
	lights[2] = Light(vec3(0, 22, -27), 10, .05, .05);
	lights[3] = Light(vec3(0, -22, 27), 10, .05, .05);
	//lights[2] = Light(vec3(10, .0, .0), 1, .1, .01);
	//lights[3] = Light(vec3(0, .0, 10.0), 1, .1, .01);
	//lights[2] = Light(vec3(1.0, 1.0, 1.0), 0.7, .1, .1);
    //lights[0] = Light(vec3(0.0, 10.0, 10.0), 0.5, 1., 1.);

    // Initialize water plane
    area_length = 10.1;
    eta = 0.9;
    d_h = 0.;
    water = WaterPlane(vec3(0, 1, 0), -9, WATER_TEX);
}

// Path tracing function
vec3 path_tracing(Ray eye_ray) {
    vec3 color = vec3(0), ncolor = vec3(0); //accumulated color and ncolor for current object
    float min_dist = INFINITY, dist, alpha = 1.;
    Hit hit;
    for(int i = 0; i < MAX_BOUNCE_NUM; i++) {
        cur_bounce_num = i;
        min_dist = INFINITY;
        // Hit water plane, hit spheres or cubes
        if(hit_objects(eye_ray, hit, false)) {
            // nothing to do
        } else {
            // intersect room return the distance between ray origin and hit spot
            intersect_cube(eye_ray, room, dist);
            hit.pos = eye_ray.origin + dist * eye_ray.dir;
            // if (eye_ray.origin.y < -water.D / water.norm.y) hit.underWater = true;
            hit.norm = -get_box_normal(hit.pos, room);
            hit.mt = get_room_mtl(hit); //different material for different face
        }
        alpha *= 1;
        ncolor = calc_light_color(hit, hit.norm, eye_ray);//calculate inner color

        color += ncolor * alpha;
    }
    return color;
}

void predict(bool val) {
    if (val) {
        gl_FragColor = vec4(0, 1, 0, 1);
    } else {
        gl_FragColor = vec4(1, 0, 0, 1);
    }
}
void main(void) {
    vec3 color = vec3(0);
    // Initialize spheres, lights and other objects
    init();

    // Fire a random ray
    rand_vec = get_rand_vec(sample_count) * 1.;//get a random vector for random sampling
    // rand_vec = get_uniform_rand_dir(sample_count) * 0.5;//get a random vector for random sampling
    vec3 ray_dir = vec3((gl_FragCoord.xy  - camera.res / 2.) / camera.res.yy * camera.fov_factor, 1);

    // vec3 ray_dir = vec3((gl_FragCoord.xy - camera.res / 2.) / camera.res.yy * camera.fov_factor, 1);
    ray_dir = normalize(trans * ray_dir);
    Ray eye_ray = Ray(ray_dir, camera.pos); // eye ray with a random deviation

    color = path_tracing(eye_ray); //calculate current frame pixel color
    vec3 prev_color = texture2D(prev_tex, gl_FragCoord.xy / camera.res).rgb;
    if (refresh == 1) {
    	// gl_FragColor = vec4(0., 0., 0., 0.);
    	gl_FragColor = vec4(color, 1.);
   	} else if (pause == 0 && hit_water) {
		gl_FragColor = vec4(color, 1.);
	} else if (sample_count < SAMPLE_THRESHOLD) { 
		gl_FragColor = vec4(mix(prev_color, color, 1./ sample_count), 1); 
		// gl_FragColor = vec4(color,1);
	} else {
		gl_FragColor = vec4(prev_color,1);
	}
	//gl_FragColor = vec4(color, 1);
}
