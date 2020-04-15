import React from 'react';
import {BASE_URL, displayErrors, ensureSignedIn, fetchMe, getAuthorization, refreshJWT} from '../Util';
import {Button, Container, Form, Grid, Header, Icon, Message, Segment} from "semantic-ui-react";
import logo from "../icon.svg";
import TopMenu from "../components/TopMenu";
import Footer from "../Footer";
import {SemanticToastContainer, toast} from "react-semantic-toasts";
import {CardElement, ElementsConsumer} from "@stripe/react-stripe-js";


class CheckoutForm extends React.Component {
  // props:
  // profile
  // stripe
  // elements
  // onSuccess

  handleSubmit = async (event) => {
    event.preventDefault();
    const {stripe, elements} = this.props;

    let response = await fetch(BASE_URL + '/api/payment-intents/', {
      method: 'POST',
      headers: {
        'Accept': 'application/json',
        'Content-Type': 'application/json',
        ...getAuthorization(),
      },
      body: JSON.stringify({
        payment_reason: 'COMPUTE_CREDITS',
      }),
    });

    let body = await response.json();
    let clientSecret = body['client_secret'];
    console.log('client secret: ', clientSecret);

    let result = await stripe.confirmCardPayment(clientSecret, {
      payment_method: {
        card: elements.getElement(CardElement),
        billing_details: {
          name: 'Jenny Rosen',
        },
      }
    });

    if (result.error) {
      console.error('Error confirming payment: ', result.error.message);
      toast(
        {
          type: 'error',
          title: 'Error Confirming Payment',
          description: <p>{result.error.message}</p>
        },
      );
    } else if (result.paymentIntent.status === 'succeeded') {
      toast(
        {
          type: 'success',
          title: 'Success',
          description: <p>You have successfully purchased 10 compute credits!</p>
        },
      );
      this.props.onSuccess();
    } else {
      toast(
        {
          type: 'error',
          title: 'Error Confirming Payment',
          description: <p>An unknown error occurred</p>
        },
      );
    }
  };

  render() {
    const {stripe} = this.props;
    return (
      <div>
        <Segment>
          <h4>Buy Compute Credits</h4>
          <p>You currently own <strong>{parseFloat(this.props.profile.compute_credits).toFixed(3)}</strong> compute credits.</p>
          <p>1 compute credit = 1 hour of GPU compute</p>
          <p style={{marginBottom: '24px'}}>Purchase 10 compute credits here for only £0.30</p>
          <form onSubmit={this.handleSubmit}>
            <CardElement/>
            <Button style={{marginTop: '10px'}} fluid color='green' type="submit" disabled={!stripe}>
              <Icon name='pound'/> Pay
            </Button>
          </form>
        </Segment>
        <Message info>
          <p>This is currently integrating with Stripes sandbox API.</p>
          <p>Enter any CVC and any future expiration date, with the following test card details:</p>
          <p><strong>4000 0025 0000 3155</strong></p>
        </Message>
      </div>
    );
  }
}

export default class Profile extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      me: {},
      password: null,
      errors: [],
      profile: {},
    }
  }

  submit = () => {
    let details = {
      id: this.state.me?.id,
      first_name: this.state.me?.first_name,
      last_name: this.state.me?.last_name,
      username: this.state.me?.username,
      email: this.state.me?.email,
    };
    if (this.state.password !== null) {
      details['password'] = this.state.password;
    }

    fetch(BASE_URL + '/api/auth/users/me/', {
      method: 'PATCH',
      headers: {
        'Accept': 'application/json',
        'Content-Type': 'application/json',
        ...getAuthorization(),
      },
      body: JSON.stringify(details),
    }).then(response => {
      if (response.status !== 200) {
        response.json().then(body => {
          this.setState({
            errors: displayErrors(
              body['first_name'],
              body['last_name'],
              body['username'],
              body['email'],
              body['password'])
          });
        });
        return;
      }

      this.setState({errors: []});
      response.json().then(body => {

        console.log(body);
      });

      toast(
        {
          type: 'success',
          title: 'Success',
          description: <p>Profile details successfully updated</p>
        },
      );
    });
  };

  keyPress = (event) => {
    if (event.keyCode === 13 && this.ableToSubmit()) {
      this.submit(event);
    }
  };

  ableToSubmit = () => {
    return this.state.firstName !== ''
      && this.state.lastName !== ''
      && this.state.username !== ''
      && this.state.email !== ''
      && this.state.password !== '';
  };

  componentDidMount = () => {
    ensureSignedIn();
    fetchMe(me => {
      this.setState({me});
    });
    this.fetchProfile();
  };

  fetchProfile() {
    fetch(BASE_URL + '/api/profiles/me/', {
      method: 'GET',
      headers: {
        'Accept': 'application/json',
        'Content-Type': 'application/json',
        ...getAuthorization(),
      },
    }).then(response => {
      if (response.status === 401) {
        refreshJWT();
        return;
      }

      if (response.status !== 200) {
        response.text().then(body => {
          console.error("Unable to get profile: ", body);
        });
        return;
      }

      response.json().then(body => {
        this.setState({profile: body});
      });
    });
  }

  render = () => (
    <div className='SitePage'>
      <TopMenu me={this.state.me}/>
      <SemanticToastContainer position='bottom-right'/>
      <Container style={{marginTop: '64px', marginBottom: '32px'}} className='SiteContents'>
        <Header as="h2" color="teal" textAlign="center">
          <img src={logo} alt="logo" className="image"/>
          Your Profile
        </Header>
        <Grid>
          <Grid.Column width={8}>
            <ElementsConsumer>
              {({stripe, elements}) => (
                <CheckoutForm
                  profile={this.state.profile}
                  stripe={stripe}
                  elements={elements}
                  onSuccess={() => this.fetchProfile()}
                />
              )}
            </ElementsConsumer>
          </Grid.Column>
          <Grid.Column width={8}>
            <Segment>
              <Form>
                <Form.Input
                  fluid
                  defaultValue={this.state.me?.first_name}
                  placeholder="First name"
                  onKeyDown={this.keyPress}
                  onChange={event => this.setState({me: {...this.state.me, first_name: event.target.value}})}
                />
                <Form.Input
                  fluid
                  defaultValue={this.state.me?.last_name}
                  placeholder="Last name"
                  onKeyDown={this.keyPress}
                  onChange={event => this.setState({me: {...this.state.me, last_name: event.target.value}})}
                />
                <Form.Input
                  fluid
                  defaultValue={this.state.me?.email}
                  icon="envelope"
                  iconPosition="left"
                  placeholder="Email"
                  onKeyDown={this.keyPress}
                  onChange={event => this.setState({me: {...this.state.me, email: event.target.value}})}
                />
                <Button
                  color="teal"
                  fluid size="large"
                  disabled={!this.ableToSubmit()}
                  onClick={this.submit}
                >
                  Save Details
                </Button>
              </Form>
            </Segment>
            <Message
              error
              header='Profile Change Unsuccessful'
              list={this.state.errors}
              hidden={this.state.errors.length === 0}
            />
          </Grid.Column>
        </Grid>
      </Container>
      {/*  </Grid.Column>*/}
      {/*</Grid>*/}
      <Footer/>
    </div>
  );
}
