$cv = Get-ItemProperty "HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion"
"BuildBranch: $($cv.BuildBranch)"
"BuildLab: $($cv.BuildLab)"
"BuildLabEx: $($cv.BuildLabEx)"
"DisplayVersion: $($cv.DisplayVersion)"
"EditionID: $($cv.EditionID)"
"CompositionEditionID: $($cv.CompositionEditionID)"
"ProductName: $($cv.ProductName)"

""
"=== Insider Status ==="
$ws = Get-ItemProperty "HKLM:\SOFTWARE\Microsoft\WindowsSelfHost\UI\Selection" -ErrorAction SilentlyContinue
if ($ws) {
    "UIContentType: $($ws.UIContentType)"
    "UIRing: $($ws.UIRing)"
    "UIBranch: $($ws.UIBranch)"
} else { "No Insider enrollment found (WindowsSelfHost UI)" }

$wa = Get-ItemProperty "HKLM:\SOFTWARE\Microsoft\WindowsSelfHost\Applicability" -ErrorAction SilentlyContinue
if ($wa) {
    "BranchName: $($wa.BranchName)"
    "ContentType: $($wa.ContentType)"
    "Ring: $($wa.Ring)"
} else { "No Insider Applicability" }

""
"=== Flighting ==="
$f = Get-ItemProperty "HKLM:\SOFTWARE\Microsoft\WindowsSelfHost\Flights" -ErrorAction SilentlyContinue
if ($f) { "Flights exist" } else { "No Flights key" }
