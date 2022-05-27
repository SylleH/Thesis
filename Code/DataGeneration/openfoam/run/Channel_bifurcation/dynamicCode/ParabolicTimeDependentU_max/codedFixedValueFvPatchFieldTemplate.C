/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Copyright (C) YEAR OpenFOAM Foundation
     \\/     M anipulation  |
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

\*---------------------------------------------------------------------------*/

#include "codedFixedValueFvPatchFieldTemplate.H"
#include "addToRunTimeSelectionTable.H"
#include "fvPatchFieldMapper.H"
#include "volFields.H"
#include "surfaceFields.H"
#include "unitConversion.H"
//{{{ begin codeInclude

//}}} end codeInclude


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

// * * * * * * * * * * * * * * * Local Functions * * * * * * * * * * * * * * //

//{{{ begin localCode

//}}} end localCode


// * * * * * * * * * * * * * * * Global Functions  * * * * * * * * * * * * * //

extern "C"
{
    // dynamicCode:
    // SHA1 = fb5cfb170e91438ca9755b551d46bb84a24ad2ea
    //
    // unique function name that can be checked if the correct library version
    // has been loaded
    void ParabolicTimeDependentU_max_fb5cfb170e91438ca9755b551d46bb84a24ad2ea(bool load)
    {
        if (load)
        {
            // code that can be explicitly executed after loading
        }
        else
        {
            // code that can be explicitly executed before unloading
        }
    }
}

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

makeRemovablePatchTypeField
(
    fvPatchVectorField,
    ParabolicTimeDependentU_maxFixedValueFvPatchVectorField
);


const char* const ParabolicTimeDependentU_maxFixedValueFvPatchVectorField::SHA1sum =
    "fb5cfb170e91438ca9755b551d46bb84a24ad2ea";


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

ParabolicTimeDependentU_maxFixedValueFvPatchVectorField::
ParabolicTimeDependentU_maxFixedValueFvPatchVectorField
(
    const fvPatch& p,
    const DimensionedField<vector, volMesh>& iF
)
:
    fixedValueFvPatchField<vector>(p, iF)
{
    if (false)
    {
        Info<<"construct ParabolicTimeDependentU_max sha1: fb5cfb170e91438ca9755b551d46bb84a24ad2ea"
            " from patch/DimensionedField\n";
    }
}


ParabolicTimeDependentU_maxFixedValueFvPatchVectorField::
ParabolicTimeDependentU_maxFixedValueFvPatchVectorField
(
    const fvPatch& p,
    const DimensionedField<vector, volMesh>& iF,
    const dictionary& dict
)
:
    fixedValueFvPatchField<vector>(p, iF, dict)
{
    if (false)
    {
        Info<<"construct ParabolicTimeDependentU_max sha1: fb5cfb170e91438ca9755b551d46bb84a24ad2ea"
            " from patch/dictionary\n";
    }
}


ParabolicTimeDependentU_maxFixedValueFvPatchVectorField::
ParabolicTimeDependentU_maxFixedValueFvPatchVectorField
(
    const ParabolicTimeDependentU_maxFixedValueFvPatchVectorField& ptf,
    const fvPatch& p,
    const DimensionedField<vector, volMesh>& iF,
    const fvPatchFieldMapper& mapper
)
:
    fixedValueFvPatchField<vector>(ptf, p, iF, mapper)
{
    if (false)
    {
        Info<<"construct ParabolicTimeDependentU_max sha1: fb5cfb170e91438ca9755b551d46bb84a24ad2ea"
            " from patch/DimensionedField/mapper\n";
    }
}


ParabolicTimeDependentU_maxFixedValueFvPatchVectorField::
ParabolicTimeDependentU_maxFixedValueFvPatchVectorField
(
    const ParabolicTimeDependentU_maxFixedValueFvPatchVectorField& ptf,
    const DimensionedField<vector, volMesh>& iF
)
:
    fixedValueFvPatchField<vector>(ptf, iF)
{
    if (false)
    {
        Info<<"construct ParabolicTimeDependentU_max sha1: fb5cfb170e91438ca9755b551d46bb84a24ad2ea "
            "as copy/DimensionedField\n";
    }
}


// * * * * * * * * * * * * * * * * Destructor  * * * * * * * * * * * * * * * //

ParabolicTimeDependentU_maxFixedValueFvPatchVectorField::
~ParabolicTimeDependentU_maxFixedValueFvPatchVectorField()
{
    if (false)
    {
        Info<<"destroy ParabolicTimeDependentU_max sha1: fb5cfb170e91438ca9755b551d46bb84a24ad2ea\n";
    }
}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

void ParabolicTimeDependentU_maxFixedValueFvPatchVectorField::updateCoeffs()
{
    if (this->updated())
    {
        return;
    }

    if (false)
    {
        Info<<"updateCoeffs ParabolicTimeDependentU_max sha1: fb5cfb170e91438ca9755b551d46bb84a24ad2ea\n";
    }

//{{{ begin code
    #line 52 "/data/channel_bifurcation/0/U/boundaryField/top"
const vectorField& Cf = patch().Cf();
		vectorField& field = *this;

		const scalar r = 0.015;
		const scalar Umax = 0.6;
		const scalar Umin = 0.02;

		scalar 	Uvar = 0.08;
		scalar 	t = this->db().time().value();
		
		if (t <= 0.35)
		{
			Uvar = 0.08+Umax*(1-(pow(t-0.175,2)/(0.175*0.175)));
		}
		if (t > 0.35 && t <= 0.6)
		{
			Uvar = 0.08+Umin*(-1+(pow(t-0.475,1)/(0.125*0.125)));
		}
		if (t > 0.6)
		{
			Uvar = 0.08;
		}
		
		forAll(Cf, faceI)
		{
			const scalar x = Cf[faceI][0];
			const scalar y = Cf[faceI][1];
			field[faceI] = vector(0, Uvar*(-1+((x*x)/(r*r))), 0);
		}
//}}} end code

    this->fixedValueFvPatchField<vector>::updateCoeffs();
}


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //

