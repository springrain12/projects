#include <iostream>
#include <papi.h>

using namespace std;

int main(void) {
    int ret = PAPI_library_init(PAPI_VER_CURRENT);

    cout << "PAPI version #"
         << PAPI_VERSION_MAJOR(ret)     << "."
         << PAPI_VERSION_MINOR(ret)     << "."
         << PAPI_VERSION_REVISION(ret)  << endl;

    return 0;
}
