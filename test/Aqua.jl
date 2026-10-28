@testitem "Aqua" tags = [:aqua] begin
    using LocalProjections
    using Aqua
    Aqua.test_all(LocalProjections)
end
