/* Golden projection coordinates from binoculars-ng 7be4f79 on one K6C frame
 * and a 3x4 flat detector: hkl_binoculars_space_qcustom_double (QX_QY_QZ,
 * QPAR_QPER on both surface orientations, uq = 0), _hkl_double and
 * _angles_double, no mask, flat 1, no limits. The resolution is tiny, so
 * indexes_0 * resolution is the coordinate binoculars computed; items are in
 * pixel order (row-major slow, fast) because nothing is masked.
 * Build and run: bin_7be4f79_projections.sh */
#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>
#include <math.h>
#include "hkl.h"
#include "hkl-binoculars.h"
#include "hkl-binoculars-private.h"

#define SLOW 3
#define FAST 4
#define N (SLOW * FAST)
#define RES 1e-12

static const double values[] = {3, 20, -30, 40, 12, 25};	/* degrees */
static const double wavelength = 1.54;
static const double cell[] = {4.1, 5.2, 6.3, 1.5, 1.6, 1.7};	/* Å, rad */
static const double euler[] = {0.1, -0.2, 0.3};			/* rad */
static const double origin[] = {0.5, -0.002, -0.0015}, e_slow[] = {0, 0, 0.001},
	e_fast[] = {0, 0.001, 0};					/* m */

/* the qcustom and hkl spaces print their matrices to stdout on every call
 * when built without NDEBUG (cglm/io.h) */
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

static void space(const char *key, HklBinocularsSpace *s, int n_axes, int first)
{
	HklBinocularsAxis *axis;
	int k = 0;
	printf("%s \"%s\": {\"names\": [", first ? "" : ",\n", key);
	darray_foreach(axis, s->axes)
		printf("%s\"%s\"", k++ ? ", " : "", g_quark_to_string(axis->name));
	printf("], \"items\": %zu, \"indexes\": [", darray_size(s->items));
	for (size_t i = 0; i < darray_size(s->items); i++) {
		printf("%s[", i ? ", " : "");
		for (int j = 0; j < n_axes; j++)
			printf("%s%td", j ? ", " : "", darray_item(s->items, i).indexes_0[j]);
		printf("]");
	}
	printf("]}");
}

int main(void)
{
	HklGeometry *g = hkl_factory_create_new_geometry(hkl_factory_get_by_name("K6C", NULL));
	hkl_geometry_axis_values_set(g, (double *)values, 6, HKL_UNIT_USER, NULL);
	hkl_geometry_wavelength_set(g, wavelength, HKL_UNIT_USER, NULL);

	HklSample *sample = hkl_sample_new("s");
	HklLattice *lattice = hkl_lattice_new(cell[0], cell[1], cell[2], cell[3], cell[4], cell[5], NULL);
	hkl_sample_lattice_set(sample, lattice);
	HklMatrix *u = hkl_matrix_new_euler(euler[0], euler[1], euler[2]);
	hkl_sample_U_set(sample, u, NULL);

	double pix[3 * N], img[N], flat[N], res[3] = {RES, RES, RES};
	size_t dims[3] = {3, SLOW, FAST};
	for (int i = 0; i < SLOW; i++)
		for (int j = 0; j < FAST; j++)
			for (int c = 0; c < 3; c++)
				pix[c * N + i * FAST + j] = origin[c] + i * e_slow[c] + j * e_fast[c];
	for (int i = 0; i < N; i++) img[i] = flat[i] = 1;

	HklBinocularsSpace *qxyz = hkl_binoculars_space_new(N, 3),
		*qpp_h = hkl_binoculars_space_new(N, 2), *qpp_v = hkl_binoculars_space_new(N, 2),
		*hkl = hkl_binoculars_space_new(N, 3), *angles = hkl_binoculars_space_new(N, 3);
	int saved = quiet();
	hkl_binoculars_space_qcustom_double(qxyz, g, img, N, 1, flat, pix, 3, dims, res, 3, NULL,
		HKL_BINOCULARS_SURFACE_ORIENTATION_HORIZONTAL, NULL, 0, 0, 0, 0,
		HKL_BINOCULARS_QCUSTOM_SUB_PROJECTION_QX_QY_QZ, 0, 0, 0, "komega", false, NULL);
	hkl_binoculars_space_qcustom_double(qpp_h, g, img, N, 1, flat, pix, 3, dims, res, 2, NULL,
		HKL_BINOCULARS_SURFACE_ORIENTATION_HORIZONTAL, NULL, 0, 0, 0, 0,
		HKL_BINOCULARS_QCUSTOM_SUB_PROJECTION_QPAR_QPER, 0, 0, 0, "komega", false, NULL);
	hkl_binoculars_space_qcustom_double(qpp_v, g, img, N, 1, flat, pix, 3, dims, res, 2, NULL,
		HKL_BINOCULARS_SURFACE_ORIENTATION_VERTICAL, NULL, 0, 0, 0, 0,
		HKL_BINOCULARS_QCUSTOM_SUB_PROJECTION_QPAR_QPER, 0, 0, 0, "komega", false, NULL);
	hkl_binoculars_space_hkl_double(hkl, g, sample, img, N, 1, flat, pix, 3, dims, res, 3, NULL,
		NULL, 0, false, NULL);
	hkl_binoculars_space_angles_double(angles, g, img, N, 1, flat, pix, 3, dims, res, 3, NULL,
		NULL, 0, "komega");
	loud(saved);

	const darray_string *names = hkl_geometry_axis_names_get(g);
	printf("{\"binoculars_version\": \"7be4f79\", \"geometry\": \"K6C\", \"axes\": [");
	for (size_t i = 0; i < darray_size(*names); i++)
		printf("%s\"%s\"", i ? ", " : "", darray_item(*names, i));
	printf("],\n \"values_deg\": ");
	row(values, 6);
	printf(", \"wavelength\": %.17g,\n \"lattice\": ", wavelength);
	row(cell, 6);
	printf(", \"u\": [");
	const HklMatrix *um = hkl_sample_U_get(sample);
	for (int i = 0; i < 3; i++) {
		double r[3] = {hkl_matrix_get(um, i, 0), hkl_matrix_get(um, i, 1), hkl_matrix_get(um, i, 2)};
		printf("%s", i ? ", " : "");
		row(r, 3);
	}
	printf("], \"ub\": [");
	const HklMatrix *ub = hkl_sample_UB_get(sample);
	for (int i = 0; i < 3; i++) {
		double r[3] = {hkl_matrix_get(ub, i, 0), hkl_matrix_get(ub, i, 1), hkl_matrix_get(ub, i, 2)};
		printf("%s", i ? ", " : "");
		row(r, 3);
	}
	printf("],\n \"shape\": [%d, %d], \"origin\": ", SLOW, FAST);
	row(origin, 3);
	printf(", \"e_slow\": ");
	row(e_slow, 3);
	printf(", \"e_fast\": ");
	row(e_fast, 3);
	printf(", \"resolution\": %.17g,\n \"spaces\": {\n", RES);
	space("qx_qy_qz", qxyz, 3, 1);
	space("qpar_qper_horizontal", qpp_h, 2, 0);
	space("qpar_qper_vertical", qpp_v, 2, 0);
	space("hkl", hkl, 3, 0);
	space("angles", angles, 3, 0);
	printf("}}\n");
	return 0;
}
