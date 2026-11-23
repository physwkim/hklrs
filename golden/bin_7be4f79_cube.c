/* Golden cubes from binoculars-ng 7be4f79's own accumulation on a synthetic
 * K6C komega scan: 20 frames, a 5x6 flat detector, an integer image, one
 * weight per frame and two masked pixels. Each frame is a qcustom QX_QY_QZ
 * space; the cubes are
 *   one_pass  hkl_binoculars_cube_new_empty, then cube_add_space per frame
 *   two_pass  hkl_binoculars_cube_new over all twenty spaces
 *   first_half, second_half  cube_add_space over frames 0-9 and 10-19
 *   merged    hkl_binoculars_cube_new_merge(first_half, second_half)
 * and each is dumped as its axes and its bins with contributions, by linear
 * offset. The weights are powers of two, so image * weight and its float sums
 * are exact. "coordinates" holds every pixel of every frame, masked or not,
 * at resolution 1e-12, for the bin-edge margin check.
 * Build and run: bin_7be4f79_cube.sh */
#include <stdio.h>
#include <stdint.h>
#include <fcntl.h>
#include <unistd.h>
#include <math.h>
#include "hkl.h"
#include "hkl-binoculars.h"
#include "hkl-binoculars-private.h"

#define FRAMES 20
#define SLOW 5
#define FAST 6
#define N (SLOW * FAST)
#define FINE 1e-12

/* mu, komega, kappa, kphi, gamma, delta in degrees; komega is scanned */
static const double values0[] = {3, 10, -30, 40, 12, 25};
static const double komega_step = 0.5;
static const double wavelength = 1.54;
static const double origin[] = {0.5, -0.00375, -0.003}, e_slow[] = {0, 0, 0.0015},
	e_fast[] = {0, 0.0015, 0};					/* m */
static const double resolutions[] = {0.0214, 0.0177, 0.0199};
static const int masked_pixels[] = {7, 22};

static double image(int f, int i) { return (7 * i + 13 * f) % 17; }
static double weight(int f) { return ldexp(1, f % 5 - 2); }

/* qcustom prints its matrices to stdout when built without NDEBUG
 * (cglm/io.h) */
static int quiet(void)
{
	fflush(stdout);
	int saved = dup(1), null = open("/dev/null", O_WRONLY);
	dup2(null, 1);
	close(null);
	return saved;
}

static void loud(int saved)
{
	fflush(stdout);
	dup2(saved, 1);
	close(saved);
}

static void row(const double *v, int n)
{
	printf("[");
	for (int j = 0; j < n; j++) printf("%s%.17g", j ? ", " : "", v[j]);
	printf("]");
}

static void space(HklBinocularsSpace *s, HklGeometry *g, int f, const double *pix,
		  const double *res, const uint8_t *mask)
{
	double img[N], flat[N];
	size_t dims[3] = {3, SLOW, FAST};
	for (int i = 0; i < N; i++) {
		img[i] = image(f, i);
		flat[i] = 1;
	}
	hkl_binoculars_space_qcustom_double(s, g, img, N, weight(f), flat, (double *)pix, 3, dims,
		(double *)res, 3, mask, HKL_BINOCULARS_SURFACE_ORIENTATION_HORIZONTAL, NULL, 0, 0, 0, 0,
		HKL_BINOCULARS_QCUSTOM_SUB_PROJECTION_QX_QY_QZ, 0, 0, 0, "komega", false, NULL);
}

static void cube(const char *key, const HklBinocularsCube *c, int first)
{
	HklBinocularsAxis *axis;
	size_t n = 1;
	int k = 0;
	printf("%s \"%s\": {\"axes\": [", first ? "" : ",\n", key);
	darray_foreach(axis, c->axes) {
		printf("%s{\"name\": \"%s\", \"resolution\": %.17g, \"imin\": %td, \"imax\": %td}",
		       k++ ? ", " : "", g_quark_to_string(axis->name), axis->resolution,
		       axis->imin, axis->imax);
		n *= axis->imax - axis->imin + 1;
	}
	printf("],\n  \"bins\": [");
	k = 0;
	for (size_t i = 0; i < n; i++)
		if (c->contributions[i])
			printf("%s[%zu, %.17g, %u]", k++ ? ", " : "", i, (double)c->photons[i],
			       c->contributions[i]);
	printf("]}");
}

int main(void)
{
	HklGeometry *g = hkl_factory_create_new_geometry(hkl_factory_get_by_name("K6C", NULL));
	hkl_geometry_wavelength_set(g, wavelength, HKL_UNIT_USER, NULL);

	double pix[3 * N], fine[3] = {FINE, FINE, FINE}, values[6];
	for (int i = 0; i < SLOW; i++)
		for (int j = 0; j < FAST; j++)
			for (int c = 0; c < 3; c++)
				pix[c * N + i * FAST + j] = origin[c] + i * e_slow[c] + j * e_fast[c];
	uint8_t mask[N] = {0};
	for (size_t m = 0; m < sizeof(masked_pixels) / sizeof(*masked_pixels); m++)
		mask[masked_pixels[m]] = 1;

	HklBinocularsSpace *spaces[FRAMES], *coords = hkl_binoculars_space_new(N, 3);
	HklBinocularsCube *one = hkl_binoculars_cube_new_empty(),
		*halves[2] = {hkl_binoculars_cube_new_empty(), hkl_binoculars_cube_new_empty()};
	ptrdiff_t at[FRAMES][N][3];
	int saved = quiet();
	for (int f = 0; f < FRAMES; f++) {
		for (int a = 0; a < 6; a++) values[a] = values0[a];
		values[1] += f * komega_step;
		hkl_geometry_axis_values_set(g, values, 6, HKL_UNIT_USER, NULL);
		spaces[f] = hkl_binoculars_space_new(N, 3);
		space(spaces[f], g, f, pix, resolutions, mask);
		hkl_binoculars_cube_add_space(one, spaces[f]);
		hkl_binoculars_cube_add_space(halves[f >= FRAMES / 2], spaces[f]);
		space(coords, g, f, pix, fine, NULL);
		for (int i = 0; i < N; i++)
			for (int c = 0; c < 3; c++)
				at[f][i][c] = darray_item(coords->items, i).indexes_0[c];
	}
	HklBinocularsCube *two = hkl_binoculars_cube_new(FRAMES, (const HklBinocularsSpace *const *)spaces);
	HklBinocularsCube *merged = hkl_binoculars_cube_new_merge(halves[0], halves[1]);
	loud(saved);

	printf("{\"binoculars_version\": \"7be4f79\", \"geometry\": \"K6C\",\n \"values_deg\": ");
	row(values0, 6);
	printf(", \"scanned\": \"komega\", \"step_deg\": %.17g, \"frames\": %d, \"wavelength\": %.17g,\n",
	       komega_step, FRAMES, wavelength);
	printf(" \"shape\": [%d, %d], \"origin\": ", SLOW, FAST);
	row(origin, 3);
	printf(", \"e_slow\": ");
	row(e_slow, 3);
	printf(", \"e_fast\": ");
	row(e_fast, 3);
	printf(",\n \"image\": \"(7 * pixel + 13 * frame) %% 17\", \"weight\": \"2^(frame %% 5 - 2)\", \"masked\": [");
	for (size_t m = 0; m < sizeof(masked_pixels) / sizeof(*masked_pixels); m++)
		printf("%s%d", m ? ", " : "", masked_pixels[m]);
	printf("],\n \"resolutions\": ");
	row(resolutions, 3);
	printf(", \"fine\": %.17g,\n \"coordinates\": [", FINE);
	for (int f = 0; f < FRAMES; f++) {
		printf("%s[", f ? ",\n  " : "");
		for (int i = 0; i < N; i++)
			printf("%s[%td, %td, %td]", i ? ", " : "", at[f][i][0], at[f][i][1], at[f][i][2]);
		printf("]");
	}
	printf("],\n \"cubes\": {\n");
	cube("one_pass", one, 1);
	cube("two_pass", two, 0);
	cube("first_half", halves[0], 0);
	cube("second_half", halves[1], 0);
	cube("merged", merged, 0);
	printf("}}\n");
	return 0;
}
