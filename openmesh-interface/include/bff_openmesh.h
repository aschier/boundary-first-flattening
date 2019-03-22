#pragma once
#include <BFFMesh.h>
#include <OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh>
#include <openmesh_doubleprecisiontraits.h>

namespace bff {
	typedef OpenMesh::TriMesh_ArrayKernelT<OpenMesh::DoublePrecisionTraits> TriMesh;
	bff::Mesh openmesh_to_bff_mesh(const TriMesh &mesh);
	TriMesh bff_mesh_to_openmesh(bff::Mesh &mesh);
}
