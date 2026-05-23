// gev_native.cpp — fast k-core and k-truss for the GEV cohesion cache.
// Replaces the pure-Python indicators. Input is a symmetric CSR adjacency
// (indptr[n+1], indices[nnz]) of the current simple graph; output is one
// int per node. k-core / k-truss are exact, deterministic quantities.
//
// Build:  g++ -O3 -march=native -shared -fPIC -o libgevnative.so gev_native.cpp
#include <vector>
#include <algorithm>
#include <cstdint>
#include <unordered_map>
using std::vector;

extern "C" {

// ---- k-core: out[v] = core number ----------------------------------------
// Batagelj & Zaversnik O(n+m) bin-sort peeling.
void kcore_csr(int n, const int* indptr, const int* indices, int* out) {
    if (n <= 0) return;
    vector<int> deg(n);
    int maxd = 0;
    for (int v = 0; v < n; ++v) { deg[v] = indptr[v + 1] - indptr[v]; if (deg[v] > maxd) maxd = deg[v]; }
    vector<int> bin(maxd + 2, 0);
    for (int v = 0; v < n; ++v) bin[deg[v]]++;
    int start = 0;
    for (int d = 0; d <= maxd; ++d) { int c = bin[d]; bin[d] = start; start += c; }
    vector<int> pos(n), vert(n);
    for (int v = 0; v < n; ++v) { pos[v] = bin[deg[v]]; vert[pos[v]] = v; bin[deg[v]]++; }
    for (int d = maxd; d >= 1; --d) bin[d] = bin[d - 1];
    bin[0] = 0;
    vector<int> core(deg);
    for (int i = 0; i < n; ++i) {
        int v = vert[i];
        for (int j = indptr[v]; j < indptr[v + 1]; ++j) {
            int u = indices[j];
            if (core[u] > core[v]) {
                int du = core[u], pu = pos[u], pw = bin[du], w = vert[pw];
                if (u != w) { vert[pu] = w; vert[pw] = u; pos[u] = pw; pos[w] = pu; }
                bin[du]++; core[u]--;
            }
        }
    }
    for (int v = 0; v < n; ++v) out[v] = core[v];
}

// ---- k-truss: out[v] = max edge-trussness over v's incident edges --------
// edge trussness via support (triangle-count) peeling; trussness = s + 2 at
// removal; an isolated node yields 0.
void truss_csr(int n, const int* indptr, const int* indices, int* out) {
    for (int v = 0; v < n; ++v) out[v] = 0;
    if (n <= 0) return;
    // per-node sorted neighbour lists (for O(log) membership / intersection)
    vector<vector<int>> adj(n);
    for (int v = 0; v < n; ++v) {
        adj[v].assign(indices + indptr[v], indices + indptr[v + 1]);
        std::sort(adj[v].begin(), adj[v].end());
    }
    // undirected edge list (u < v) + id map
    vector<int> eu, ev;
    std::unordered_map<int64_t, int> eid;
    eid.reserve(indptr[n]);
    for (int u = 0; u < n; ++u)
        for (int w : adj[u])
            if (w > u) { eid[(int64_t)u * n + w] = (int)eu.size(); eu.push_back(u); ev.push_back(w); }
    int m = (int)eu.size();
    if (m == 0) return;
    auto edge_of = [&](int a, int b) -> int {
        if (a > b) std::swap(a, b);
        auto it = eid.find((int64_t)a * n + b);
        return it == eid.end() ? -1 : it->second;
    };
    // support(e) = number of triangles containing e
    vector<int> sup(m, 0);
    int maxs = 0;
    for (int e = 0; e < m; ++e) {
        const vector<int>& A = adj[eu[e]];
        const vector<int>& B = adj[ev[e]];
        int i = 0, j = 0, s = 0;
        while (i < (int)A.size() && j < (int)B.size()) {
            if (A[i] == B[j]) { ++s; ++i; ++j; }
            else if (A[i] < B[j]) ++i; else ++j;
        }
        sup[e] = s; if (s > maxs) maxs = s;
    }
    // bucket peel by current support
    vector<vector<int>> bucket(maxs + 1);
    for (int e = 0; e < m; ++e) bucket[sup[e]].push_back(e);
    vector<char> removed(m, 0);
    vector<int> truss(m, 0);
    int k = 2, done = 0, cur = 0;
    while (done < m) {
        while (cur <= maxs && bucket[cur].empty()) ++cur;
        if (cur > maxs) break;
        int e = bucket[cur].back(); bucket[cur].pop_back();
        if (removed[e] || sup[e] != cur) continue;
        if (sup[e] + 2 > k) k = sup[e] + 2;
        truss[e] = k; removed[e] = 1; ++done;
        int u = eu[e], v = ev[e];
        const vector<int>& A = adj[u];
        const vector<int>& B = adj[v];
        int i = 0, j = 0;
        while (i < (int)A.size() && j < (int)B.size()) {  // common neighbours w
            if (A[i] == B[j]) {
                int w = A[i]; ++i; ++j;
                int e1 = edge_of(u, w), e2 = edge_of(v, w);
                if (e1 < 0 || e2 < 0 || removed[e1] || removed[e2]) continue;
                for (int ex : {e1, e2}) {
                    if (sup[ex] > cur) { sup[ex]--; bucket[sup[ex]].push_back(ex);
                                         if (sup[ex] < cur) cur = sup[ex]; }
                }
            } else if (A[i] < B[j]) ++i; else ++j;
        }
    }
    // node trussness = max over incident edges
    for (int e = 0; e < m; ++e) {
        if (truss[e] > out[eu[e]]) out[eu[e]] = truss[e];
        if (truss[e] > out[ev[e]]) out[ev[e]] = truss[e];
    }
}

}  // extern "C"
