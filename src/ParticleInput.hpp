#pragma once

#include "Parts.hpp"

#include <cctype>
#include <cstdio>
#include <fstream>
#include <locale>
#include <sstream>
#include <string>

template <class S, int PD, int SD>
struct ParticleInput {
    std::array<Vector<S>, PD> position, velocity, acceleration;
    std::array<Vector<S>, SD> strength;
    Vector<S> radius;

    bool read(const char* path, bool allow_kinematics) {
        std::ifstream file(path);
        if (!file) return report(path, 0, "cannot open particle input");

        constexpr size_t base_width = PD+SD+1;
        std::array<S, base_width+2*PD> values;
        size_t width = 0;
        size_t line_number = 0;
        std::string line;
        while (std::getline(file, line)) {
            ++line_number;
            std::istringstream row(line);
            row.imbue(std::locale::classic());
            size_t count = 0;
            while (row >> std::ws && !row.eof()) {
                if (count == values.size()) return report(path, line_number, "too many columns");
                S value;
                if (!(row >> value) || !std::isfinite(value) ||
                    (!row.eof() && !std::isspace(static_cast<unsigned char>(row.peek())))) {
                    return report(path, line_number, "expected a finite number");
                }
                values[count++] = value;
            }
            if (count == 0) continue;
            if (width == 0) {
                if (count != base_width &&
                    !(allow_kinematics && (count == base_width+PD || count == base_width+2*PD))) {
                    return report(path, line_number, "unsupported column count");
                }
                width = count;
            } else if (count != width) {
                return report(path, line_number, "inconsistent column count");
            }

            for (int d=0; d<PD; ++d) position[d].push_back(values[d]);
            if (width >= base_width+PD) {
                for (int d=0; d<PD; ++d) velocity[d].push_back(values[PD+d]);
            }
            if (width == base_width+2*PD) {
                for (int d=0; d<PD; ++d) acceleration[d].push_back(values[2*PD+d]);
            }
            for (int d=0; d<SD; ++d) strength[d].push_back(values[width-SD-1+d]);
            radius.push_back(values[width-1]);
        }
        if (file.bad() || !file.eof()) return report(path, 0, "error reading particle input");
        if (radius.empty()) return report(path, 0, "no particles found");
        return true;
    }

    template <class A, int OD>
    void move_to_sources(Parts<S,A,PD,SD,OD>& parts) {
        const size_t count = radius.size();
        parts.x.swap(position);
        parts.s.swap(strength);
        parts.r.swap(radius);
        parts.resize(count);
    }

private:
    static bool report(const char* path, size_t line, const char* message) {
        if (line != 0) std::fprintf(stderr, "%s:%zu: %s\n", path, line, message);
        else std::fprintf(stderr, "%s: %s\n", path, message);
        return false;
    }
};