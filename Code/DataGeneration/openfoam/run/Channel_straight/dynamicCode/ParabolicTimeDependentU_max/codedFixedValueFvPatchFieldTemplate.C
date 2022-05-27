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
    // SHA1 = f5f5973c5663f38b39eae53c42d8b979ff49a825
    //
    // unique function name that can be checked if the correct library version
    // has been loaded
    void ParabolicTimeDependentU_max_f5f5973c5663f38b39eae53c42d8b979ff49a825(bool load)
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
    "f5f5973c5663f38b39eae53c42d8b979ff49a825";


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
        Info<<"construct ParabolicTimeDependentU_max sha1: f5f5973c5663f38b39eae53c42d8b979ff49a825"
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
        Info<<"construct ParabolicTimeDependentU_max sha1: f5f5973c5663f38b39eae53c42d8b979ff49a825"
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
        Info<<"construct ParabolicTimeDependentU_max sha1: f5f5973c5663f38b39eae53c42d8b979ff49a825"
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
        Info<<"construct ParabolicTimeDependentU_max sha1: f5f5973c5663f38b39eae53c42d8b979ff49a825 "
            "as copy/DimensionedField\n";
    }
}


// * * * * * * * * * * * * * * * * Destructor  * * * * * * * * * * * * * * * //

ParabolicTimeDependentU_maxFixedValueFvPatchVectorField::
~ParabolicTimeDependentU_maxFixedValueFvPatchVectorField()
{
    if (false)
    {
        Info<<"destroy ParabolicTimeDependentU_max sha1: f5f5973c5663f38b39eae53c42d8b979ff49a825\n";
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
        Info<<"updateCoeffs ParabolicTimeDependentU_max sha1: f5f5973c5663f38b39eae53c42d8b979ff49a825\n";
    }

//{{{ begin code
    #line 51 "//data/0/U/boundaryField/top"
const vectorField& Cf = patch().Cf();
		vectorField& field = *this;

		const scalar r = 13.0*1e-3;
		const scalar Umax = 0.4;
		const scalar Umin = 0.02;

		scalar 	Uvar = 0.08;
		scalar 	t = this->db().time().value();
		scalar	tau = fmod(t,1);
		
		if (tau > 0 && tau <= 0.35)
		{
			Uvar = 0.08+Umax*(1-(pow(tau-0.175,2)/(0.175*0.175)));
		}
		if (tau > 0.35 && tau <= 0.6)
		{
			Uvar = 0.08+Umin*(-1+(pow(tau-0.475,2)/(0.125*0.125)));
		}
		if (tau > 0.6 && tau <= 1)
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

