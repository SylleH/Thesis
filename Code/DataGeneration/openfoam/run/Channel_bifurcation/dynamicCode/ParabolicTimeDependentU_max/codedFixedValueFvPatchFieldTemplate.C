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
    // SHA1 = 09634cad939d8419607bf6e666e01e429b4f8f28
    //
    // unique function name that can be checked if the correct library version
    // has been loaded
    void ParabolicTimeDependentU_max_09634cad939d8419607bf6e666e01e429b4f8f28(bool load)
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
    "09634cad939d8419607bf6e666e01e429b4f8f28";


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
        Info<<"construct ParabolicTimeDependentU_max sha1: 09634cad939d8419607bf6e666e01e429b4f8f28"
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
        Info<<"construct ParabolicTimeDependentU_max sha1: 09634cad939d8419607bf6e666e01e429b4f8f28"
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
        Info<<"construct ParabolicTimeDependentU_max sha1: 09634cad939d8419607bf6e666e01e429b4f8f28"
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
        Info<<"construct ParabolicTimeDependentU_max sha1: 09634cad939d8419607bf6e666e01e429b4f8f28 "
            "as copy/DimensionedField\n";
    }
}


// * * * * * * * * * * * * * * * * Destructor  * * * * * * * * * * * * * * * //

ParabolicTimeDependentU_maxFixedValueFvPatchVectorField::
~ParabolicTimeDependentU_maxFixedValueFvPatchVectorField()
{
    if (false)
    {
        Info<<"destroy ParabolicTimeDependentU_max sha1: 09634cad939d8419607bf6e666e01e429b4f8f28\n";
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
        Info<<"updateCoeffs ParabolicTimeDependentU_max sha1: 09634cad939d8419607bf6e666e01e429b4f8f28\n";
    }

//{{{ begin code
    #line 51 "/data/run/channel_bifurcation/0/U/boundaryField/top"
const vectorField& Cf = patch().Cf();
		vectorField& field = *this;

		const scalar r = 10.0*1e-3;
		const scalar rcm = r*100;
		scalar 	t = this->db().time().value();
		scalar	tau = fmod(t,1);
		scalar	Uvar = 0.08;
		
   		const scalar a0    = 97.4;
scalar a[14]={-6.12644665,-42.59636182,-0.77437875,10.78546692,-6.90775766,-1.1372423,2.37198109,-0.7100494,0.39921836,1.42133022,2.57732891,-1.19777814,-2.20580923,1.37967485};

   		scalar b[14]={-48.95362598,-2.15308961,26.22974423,-3.63331624,-3.37693086,6.26447831,-0.08230374,0.14968907,1.30674276,0.74004953,-1.46645236,-3.36070789,1.00422553,0.08805414};
   		scalar Q = 0.5*a0;
   		const scalar t_min = 0;
   		const scalar t_max = t_min + 1.0;
   		scalar n = M_PI * ( 2 * ( tau-t_min ) / ( t_max - t_min ) - 1 );
		
		if (tau > 0 && tau <= 1)
		{
			for (int i = 0; i < 10; i++)
			{
   			Q += ( a[i]*cos((i+1)*n) + b[i]*sin((i+1)*n) );
   			}
			Uvar = Q/(M_PI*(rcm*rcm));
			Uvar = Uvar/100;
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

