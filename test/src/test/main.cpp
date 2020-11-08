#include "test/pch.h"

TEST(gtest, test) {
    ASSERT_TRUE(true);
}

int main() {
    testing::InitGoogleTest();
    return RUN_ALL_TESTS();
}